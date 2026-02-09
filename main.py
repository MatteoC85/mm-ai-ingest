import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "status": "healthy",
        "build": os.environ.get("BUILD_ID", "no-build-id")
    }
