# Copyright 2025 Santiago Gallego
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io

import yaml
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import CONFIG_PATH, MAX_FILE_SIZE, SUPPORTED_IMAGE_FORMATS, SUPPORTED_MODELS
from app.core.pipeline import STDRPipeline


async def get_image(image_file: UploadFile) -> Image.Image:
    """Validates and loads an uploaded file into a PIL Image."""

    if image_file.content_type not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(status_code=400, detail="Only BMP, JPEG, PNG, and WEBP formats are supported.")

    if image_file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Limit is {MAX_FILE_SIZE / 1024 / 1024:.2f} MB.",
        )

    image_bytes = await image_file.read()

    try:
        with io.BytesIO(image_bytes) as buffer:
            image = Image.open(buffer).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="File provided is not a valid image.") from e

    return image


class GetModel:
    """Preloads and provides ML models as a dependency."""

    def __init__(self):
        self.models = {}
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.models["craft_parseq"] = STDRPipeline(config)

    def __call__(self, name: str = "craft_parseq"):
        # NOTE: Use this exception when the client can select the model (i.e. passing the model's name as a query parameter).
        # if name not in SUPPORTED_MODELS:
        #     raise HTTPException(status_code=400, detail=f"{name} is not a valid model.")

        model = self.models.get(name)

        if model is None:
            raise HTTPException(status_code=503, detail=f"Model {name} is not loaded.")

        return model


get_model = GetModel()
