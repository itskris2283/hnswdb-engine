# Push & Publication Checklist for HNSWDB

This checklist guides publishing the repository to GitHub with complete verification and honesty.

---

## 1. Pre-Push Verification (Automated)

Ensure all local verification passes cleanly on the repository:
```bash
# 1. Verify claims and consistency checker
python scripts/check_claims.py

# 2. Verify all markdown documentation links
python scripts/check_docs_links.py

# 3. Verify README REST API examples and Python quickstart
python scripts/verify_readme.py

# 4. Verify C++ tests
ctest --test-dir build/rel -C Release --output-on-failure

# 5. Verify Python tests
pytest tests/python/
```

---

## 2. GitHub Repository Creation & Push

1. **Create GitHub Repository:**
   - Go to GitHub and create a new public repository named `hnswdb`.
   - Do **not** initialize with a README, .gitignore, or license (already present in repo).

2. **Configure Remote & Push:**
   ```bash
   git remote add origin https://github.com/<your-username>/hnswdb.git
   git branch -M main
   git push -u origin main
   git push --tags
   ```

---

## 3. GitHub Actions CI & Badges

1. **Check CI Execution:**
   - Navigate to the **Actions** tab on your GitHub repository.
   - Confirm that the `.github/workflows/ci.yml` workflow runs and passes cleanly on Ubuntu / Windows.

2. **Add CI Badge (Only After Verification):**
   - Record the passing run URL in `results/raw/ci_runs.txt`.
   - Once verified green, optionally insert the CI status badge at the top of `README.md`:
     ```markdown
     [![CI](https://github.com/<your-username>/hnswdb/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/hnswdb/actions/workflows/ci.yml)
     ```

---

## 4. Repository Presentation & Topics

1. **Pin Repository:**
   - Pin `hnswdb` to your GitHub profile overview.

2. **Add Repository Topics:**
   - Under the repository "About" settings, add the following topics:
     - `vector-database`
     - `hnsw`
     - `cpp`
     - `ann`
     - `simd`
     - `pybind11`

3. **Verify Documentation Links:**
   - Confirm that `docs/INVESTIGATIONS.md` is directly linked from `README.md` (under "Status and Limitations").
