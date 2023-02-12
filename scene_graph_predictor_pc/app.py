from uuid import uuid4
from pathlib import Path
from typing import Optional, List

import aiofiles
import trimesh
from fastapi import FastAPI, UploadFile, Query
from PIL import Image

from .inference import SceneGraphPredictor

app = FastAPI()
model: Optional[SceneGraphPredictor] = None


def get_model() -> SceneGraphPredictor:
    global model
    if model is None:
        model = SceneGraphPredictor()

    return model


@app.get("/")
async def root():
    return {"message": "Hello World"}


async def save_upload_file(upload_file: UploadFile) -> str:
    file_path = f"uploaded_images/{uuid4()}{Path(upload_file.filename).suffix}"
    async with aiofiles.open(file_path, "wb") as f:
        while content := await upload_file.read(4 * 1024):
            await f.write(content)

    return file_path


@app.post("/inference")
async def inference(
    plydata: UploadFile, topk: int
):
    ply_path = await save_upload_file(plydata)
    ply_data = trimesh.load(ply_path, process=False)

    model = get_model()
    res = model.inference(ply_data, topk)

    return {"triplets": res}