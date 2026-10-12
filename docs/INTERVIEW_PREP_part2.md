# HNSWDB: Technical Interview Preparation (Part 2)

Continued from [Part 1](INTERVIEW_PREP_part1.md).

---

## 4. Technical Q&As (Q4-Q12)

### Q4: "How are deletes and tombstones handled, and what are the trade-offs?"
> "HNSWDB implements soft deletion using a `deleted_` byte array (`uint8_t` per node). Deletions mark the tombstone in $O(1)$ and decrement `live_count_`. During query traversal, tombstones still act as routing bridges across the graph to maintain connectivity, but are filtered out from the final top-$k$ result set. The trade-off is memory: tombstoned nodes continue occupying link slots and vector memory until `compact()` rebuilds the graph."

### Q5: "Explain your concurrency model and how sorted lock acquisition prevents deadlocks."
> "HNSWDB uses a two-tier synchronization design:
> 1. A coarse `std::shared_mutex structure_lock_`: reader searches take shared read locks, while structural operations (`resize`, `compact`) take exclusive write locks.
> 2. Fine-grained striped spinlocks: 1,024 spinlocks (`striped_locks_`). When adding a bidirectional edge between nodes $u$ and $v$, locks are acquired in strictly ascending order: $\min(u, v) \pmod{1024}$ first, then $\max(u, v) \pmod{1024}$. This global lock ordering mathematically guarantees cycle-free lock acquisition, eliminating deadlocks."

### Q6: "Why do you fall back to exact brute force for selective metadata filters, and how is selectivityHits estimated?"
> "Post-filtering on graphs fails when predicate selectivity is high (e.g. only 1 in 1,000 vectors match), because the beam search terminates before encountering enough matching candidates. Pre-filtering by traversing only matching nodes fragments the graph, trapping search in disconnected subgraphs. HNSWDB uses an inverted postings index to estimate predicate selectivity: if matching vector count is small, the engine falls back to an exact SIMD brute-force scan over the matching subset, guaranteeing exact recall and high throughput."

### Q7: "What is the difference between CRC32 checksum verification and atomic write-rename?"
> "They protect against two distinct failure modes:
> - **CRC32 Checksum (IEEE 802.3):** Protects against data corruption and silent bit rot (e.g. disk sector failure, incomplete writes). If any byte in the header or payload changes, the trailer CRC check fails on load, raising `CorruptFile`.
> - **Atomic Write-Rename:** Protects against application or system crashes mid-write. Data is written to a temporary file (`.hdb.tmp`) and flushed to disk. Once fully written, an OS atomic rename overwrites the destination path, ensuring that the target file is never left in a truncated, half-written state."

### Q8: "What is the memory layout of HNSWDB, and how many bytes does it use per vector?"
> "HNSWDB uses a split flat array layout:
> - Vectors: contiguous `float` buffer = $dim \times 4$ bytes.
> - Layer 0 links: contiguous `uint32_t` buffer = $M_0 \times 4 = 2M \times 4$ bytes per vector.
> - Upper layer links: average $M \times 4 \times \sum_{l=1}^{\infty} e^{-l \ln M} \approx \frac{M}{M-1} \times 4$ bytes $\approx 4$ bytes.
> - Metadata & tracking: label array ($8$ bytes), deleted flag ($1$ byte).
>
> For $M=16, dim=64$: Vector ($256$ B) + Layer 0 ($128$ B) + Upper layers (~$4$ B) + Labels ($8$ B) + Flags ($1$ B) $\approx 397$ bytes/vector."

### Q9: "Why are equal-recall comparisons essential when benchmarking ANN engines?"
> "Raw QPS numbers without controlled recall are meaningless: any engine can achieve hundreds of thousands of QPS by setting $ef_{search}=1$, but recall will be unacceptably low. Conversely, high recall can be achieved by setting $ef_{search}=1000$ at the cost of high latency. An equal-recall comparison fixes the target accuracy (e.g. Recall@10 = 0.99) and interpolates the exact QPS and latency required by each engine to hit that target, providing a fair, apples-to-apples performance comparison."

### Q10: "What are HNSWDB's architectural limits, and how would you add SQ8, PQ, mmap, and sharding?"
> - **Current Limits:** In-memory only (limited by RAM), unquantized 32-bit floats, single-node capacity.
> - **Scalar Quantization (SQ8):** Quantize each float32 coordinate to uint8 using per-dimension min/max bounds, cutting vector memory by a factor of 4. Compute distances using AVX2 `_mm256_maddubs_epi16` integer instructions.
> - **Product Quantization (PQ):** Decompose vectors into sub-vectors and map to centroid codebooks, reducing 128d vectors to 16 bytes.
> - **Memory-Mapped Storage (mmap):** Align the `.hdb` binary format to 4 KB page boundaries and use `mmap` with `MADV_RANDOM`, allowing the OS page cache to evict rarely-visited Layer 0 nodes under memory pressure.
> - **Distributed Sharding:** Implement consistent hashing across partition keys with a two-tier scatter-gather coordinator or Raft consensus group."

### Q11: "Where does HNSWDB lose against FAISS or hnswlib, and what was measured?"
> "HNSWDB loses in two measured areas:
> 1. **Single-Thread Build Time on Low Dimensions:** On low dimensions (32d, 64d), hnswlib builds faster. See the work-counting table in [INVESTIGATIONS.md](INVESTIGATIONS.md).
> 2. **Tightly-Clustered Data at Low $ef_c$:** A small number of nodes remain unreachable on tightly clustered data at low $ef_c$. See target T5 in [REPORT.md](../REPORT.md) and Story 3."

### Q12: "Explain the clustered-data finding and your ablation study outcome."
> "See Story 3 above and the detailed ablation analysis in [INVESTIGATIONS.md](INVESTIGATIONS.md). A 25-seed matrix and ablation study traced the recall gap to the original construction defaults (new_node_degree='M0', keep_pruned=true). Switching to reference-style semantics (new_node_degree='M', keep_pruned=false) closed the recall gap and substantially cut build time."

---

## 5. Claims NOT to Make in an Interview

Be skeptical of your own system. Never make these claims:

| Forbidden Claim | Why It Is Erroneous | What to Say Instead |
| :--- | :--- | :--- |
| **"Claiming unconditional superiority over FAISS"** | FAISS is heavily optimized in C++ and achieves faster index build times on low dimensions. | "HNSWDB achieves competitive parity on Fashion-MNIST search recall and latency, but FAISS builds faster on low dimensions." |
| **"Claiming the engine is entirely non-blocking"** | The engine uses 1,024 striped spinlocks and a shared mutex; it is fine-grained locking, not lockless. | "We use fine-grained striped spinlocks with sorted lock acquisition to avoid deadlocks." |
| **"Claiming the recall gap is an inherent algorithmic limitation of HNSW"** | False: hnswlib achieves high recall on the identical dataset at $ef_c=100$. | "It is an implementation-specific construction weakness related to candidate degree allocation and heuristic pruning." |
| **"Claiming ThreadSanitizer verified cleanliness on Windows"** | The local development machine is Windows MSVC where TSan cannot run. | "Linux sanitizer workflows (ASan, UBSan, TSan) are configured in GitHub Actions CI; local host status is honestly reported as NOT RUN." |
| **"Claiming 3 million duplicate links existed"** | Mathematically impossible in a 5,000-node graph ($5000 \times 32 = 160,000$ maximum links). | "Invariant unit tests verify 0 duplicate links exist in snapshot graphs; our ablation study confirmed dedup had no measurable effect on recall." |
| **"Claiming HNSWDB beats hnswlib or FAISS on recall"** | False: the differences are within noise (e.g. the T3 gap is smaller than hnswlib's standard deviation across the 25-build matrix). | "HNSWDB and reference baselines achieve recall parity; minor differences are within random seed noise." |
| **"Claiming HNSWDB has no unreachable nodes"** | False: on tightly clustered synthetic data, HNSWDB averages 9.32 unreachable nodes per index under 1-NN self-retrieval (target T5 is an honest MISS). | "HNSWDB averages 9.32 unreachable nodes on Dataset B; baselines also exhibit unreachable nodes under self-retrieval." |

---

## 6. How to Describe the Verification Pipeline

Say: "Benchmark tables are rendered from raw result files by scripts, and automated checks fail if a table is hand-edited, a banned claim appears, or the frozen target file changes. It is a local consistency guard, not a security mechanism."

Do not say: "cryptographic", "audited", or that the targets are tamper-proof. The freeze is a git tag plus a script in the same repository.

---

## 7. Script: "Tell Me About a Time You Were Wrong"

> *"When investigating our recall plateau on tightly-clustered synthetic data ($\sigma=0.1$), I initially hypothesized that duplicate link insertions during edge backfilling were causing graph connectivity loss. I implemented candidate deduplication and convinced myself it was the solution, even drafting notes attributing recall recovery to this fix.*
>
> *However, when I stepped back to apply rigorous systems thinking, I realized my hypothesis had not been tested. First, I derived the mathematical invariants: in our 5,000-node graph with $M_0=32$, layer 0 contains at most 160,000 links total, so millions of duplicate links in a snapshot graph was mathematically impossible. Second, I wrote an automated C++ invariant test that checked every link in the graph, which proved that snapshot graphs already had zero duplicate links.*
>
> *Third, I designed a multi-seed ablation benchmark comparing the engine with and without the deduplication logic across 5 random seeds: the recall delta was zero (pure noise). My hypothesis had been completely falsified.*
>
> *I immediately retracted the claim from our technical documentation, introduced strict invariant tests to catch invalid assertions, and redesigned our diagnostic approach. By writing 1-NN self-retrieval diagnostics and a systematic ablation matrix, we traced the issue to original construction defaults (`new_node_degree="M0"`, `keep_pruned=true`), as detailed in Story 3 and INVESTIGATIONS.md.*
>
> *That experience reinforced an essential systems engineering principle: **never confuse a plausible narrative with an isolated, verified cause. Write invariant assertions before writing fixes, and measure before concluding.**"*
