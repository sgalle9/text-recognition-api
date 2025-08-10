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

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from PIL.Image import Image

from app.config import MAX_SIZE
from app.core.utils import resize_bbox
from app.dependencies import get_image, get_model
from app.schemas import PostprocessingParams, PredictionItem

router = APIRouter(prefix="/v1")


@router.post("/predict", response_model=list[PredictionItem])
async def predict(
    image: Annotated[Image, Depends(get_image)], postprocess_params: Annotated[PostprocessingParams, Query()]
):
    """Receives an image, performs text detection and recognition, and returns the prediction as JSON."""

    w, h = image.size
    image.thumbnail((MAX_SIZE, MAX_SIZE))
    new_w, new_h = image.size
    ratio_x, ratio_y = (w / new_w), (h / new_h)

    model = get_model()
    prediction = model(image, **postprocess_params.model_dump())

    prediction = [
        {"text": text, "confidence": word_proba, "bounding_box": resize_bbox(bbox, ratio_x, ratio_y).tolist()}
        for text, word_proba, bbox in prediction
    ]

    return prediction
