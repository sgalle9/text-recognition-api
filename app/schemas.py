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

from pydantic import BaseModel, Field

from app.config import MAX_SIZE


class PostprocessingParams(BaseModel):
    """Query parameters used to fine-tune the behavior of the text detection and bounding box generation algorithm."""

    text_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        title="Text Threshold",
        description="**Increase** this value to filter out noise, background patterns, and other non-text elements. **Decrease** it to improve the detection of faint, blurry, or small text that the model is missing.",
    )
    link_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        title="Link Threshold",
        description="**Increase** this value if separate words are being incorrectly merged. **Decrease** it if single words are being incorrectly split into multiple parts.",
    )

    low_text: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        title="Text Boundary Threshold",
        description="**Increase** this value for tighter, more conservative bounding boxes. **Decrease** it to increase the bounding box area.",
    )
    area_threshold: int = Field(
        default=30,
        ge=0,
        le=MAX_SIZE**2,
        title="Minimum Area Threshold",
        description="Filters out any detected region whose bounding box area is smaller than this value. This value is in pixels.",
    )

    model_config = {
        "extra": "forbid"  # Forbid extra parameters. If a client tries to send some extray data in the query parameters, they will receive an error response.
    }


class PredictionItem(BaseModel):
    text: str = Field(title="Recognized Text")
    confidence: float = Field(ge=0.0, le=1.0, title="Recognition Confidence Score")
    bounding_box: list[tuple[float, float]] = Field(min_length=4, max_length=4, title="Bounding Box Coordinates")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Hello",
                    "confidence": 0.98,
                    "bounding_box": [[87.8, 135.8], [487.4, 121.8], [489.2, 171.3], [89.6, 185.3]],
                }
            ]
        }
    }
