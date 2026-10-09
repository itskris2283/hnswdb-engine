import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
import numpy as np
from pydantic import BaseModel, Field

from hnswdb import HNSWIndex, HnswdbError

app = FastAPI(title="HNSWDB REST API", version="0.1.0")

collections: Dict[str, HNSWIndex] = {}


class CreateCollectionRequest(BaseModel):
    dim: int = Field(..., gt=0, le=4096)
    metric: str = "l2"
    max_elements: int = 100000
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    seed: int = 42


class InsertRequest(BaseModel):
    vectors: List[List[float]]
    ids: List[int]
    metadata: Optional[List[Dict[str, Any]]] = None


class QueryRequest(BaseModel):
    vector: List[float]
    k: int = 10
    ef_search: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None


class DeleteRequest(BaseModel):
    ids: List[int]


class QueryResponse(BaseModel):
    ids: List[int]
    distances: List[float]
    metadata: List[Dict[str, Any]]
    took_ms: float


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/collections")
def list_collections() -> Dict[str, List[str]]:
    return {"collections": list(collections.keys())}


@app.post("/collections/{name}", status_code=status.HTTP_201_CREATED)
def create_collection(name: str, req: CreateCollectionRequest) -> Dict[str, Any]:
    if name in collections:
        raise HTTPException(status_code=400, detail=f"Collection '{name}' already exists")
    try:
        idx = HNSWIndex(
            dim=req.dim,
            metric=req.metric,
            max_elements=req.max_elements,
            M=req.M,
            ef_construction=req.ef_construction,
            ef_search=req.ef_search,
            seed=req.seed,
        )
        collections[name] = idx
        return {
            "status": "created",
            "name": name,
            "dim": req.dim,
            "metric": req.metric,
            "max_elements": req.max_elements,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/collections/{name}")
def delete_collection(name: str) -> Dict[str, str]:
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    del collections[name]
    return {"status": "deleted", "name": name}


@app.post("/collections/{name}/insert")
def insert_vectors(name: str, req: InsertRequest) -> Dict[str, Any]:
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    idx = collections[name]
    if len(req.vectors) != len(req.ids):
        raise HTTPException(status_code=400, detail="Length of vectors and ids must match")

    try:
        vecs_arr = np.array(req.vectors, dtype=np.float32)
        ids_arr = np.array(req.ids, dtype=np.int64)
        idx.add(vecs_arr, ids_arr, metadata=req.metadata)
        return {"status": "inserted", "count": len(req.ids), "total_size": len(idx)}
    except HnswdbError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/{name}/query", response_model=QueryResponse)
def query_collection(name: str, req: QueryRequest) -> QueryResponse:
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    idx = collections[name]

    try:
        q_arr = np.array(req.vector, dtype=np.float32)
        t0 = time.perf_counter()
        ids_res, dists_res = idx.search(q_arr, k=req.k, ef_search=req.ef_search, filter=req.filter)
        took_ms = (time.perf_counter() - t0) * 1000.0

        res_ids = ids_res[0].tolist()
        res_dists = dists_res[0].tolist()

        res_meta = []
        for fid in res_ids:
            if fid != -1 and idx.contains(fid):
                res_meta.append(idx.get_metadata(fid))
            else:
                res_meta.append({})

        return QueryResponse(
            ids=res_ids,
            distances=res_dists,
            metadata=res_meta,
            took_ms=round(took_ms, 4),
        )
    except HnswdbError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{name}/vectors")
def delete_vectors(name: str, req: DeleteRequest) -> Dict[str, Any]:
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    idx = collections[name]
    try:
        deleted_count = idx.delete(req.ids)
        return {"status": "deleted", "deleted_count": deleted_count, "remaining_size": len(idx)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{name}/stats")
def get_stats(name: str) -> Dict[str, Any]:
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    idx = collections[name]
    return idx.stats()


@app.post("/collections/{name}/save")
def save_collection(name: str, path: str = Query(...)) -> Dict[str, str]:
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    idx = collections[name]
    try:
        idx.save(path)
        return {"status": "saved", "name": name, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/{name}/load")
def load_collection(name: str, path: str = Query(...)) -> Dict[str, Any]:
    try:
        idx = HNSWIndex.load(path)
        collections[name] = idx
        return {"status": "loaded", "name": name, "path": path, "size": len(idx), "dim": idx.dim()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
