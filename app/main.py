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

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.dependencies import get_model
from app.routers import demo, predict

app = FastAPI(
    title="Scene Text Detection and Recognition API",
    description="An API for detecting and recognizing text in natural images.",
    version="1.0.0",
)

app.include_router(predict.router)

app = gr.mount_gradio_app(app, demo.gradio_app, path="/demo")


@app.get("/")
async def main():
    # return {"message": "Welcome to the Scene Text Detection and Recognition API. Navigate to /docs for the API documentation."}
    # NOTE: Redirecting to Gradio endpoint (for deployment on HuggingFace Spaces).
    return RedirectResponse("/demo")


@app.get("/health")
async def health():
    """Health check endpoint."""
    status = "ok" if len(get_model.models) > 0 else "unavailable"
    return {"status": status}
