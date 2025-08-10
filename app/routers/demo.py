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

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

from app.config import MAX_SIZE
from app.core.utils import plot_predictions
from app.dependencies import get_model


def predict_and_plot(image: Image.Image) -> Image.Image:
    """Receives an image, performs text detection and recognition, and returns the annotated image."""

    image.thumbnail((MAX_SIZE, MAX_SIZE))

    model = get_model()
    predictions = model(image)

    # Convert the Matplotlib figure into a PIL image.
    fig = plot_predictions(image, predictions)
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    output_image = Image.open(image_buffer)

    return output_image


inputs = gr.Image(type="pil", label="Input Image")

outputs = gr.Image(type="pil", label="Annotated Image")

DESCRIPTION = """
## How to Use
1. **Upload** an image, or click on the **example** below.
2. Click **Submit** to run the detection.
3. Click **Clear** to try a new image.

The tool will highlight all detected text right on your image.
"""

gradio_app = gr.Interface(
    fn=predict_and_plot,
    inputs=inputs,
    outputs=outputs,
    examples=["./assets/example.jpg"],
    title="Scene Text Detection and Recognition Demo",
    description=DESCRIPTION,
    flagging_mode="never",
)
