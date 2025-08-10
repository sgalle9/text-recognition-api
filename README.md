---
title: Text Detection and Recognition Api
emoji: ⚡
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---

# Scene Text Detection and Recognition API

This project provides an API for scene text detection and recognition. It utilizes the [CRAFT](https://github.com/clovaai/CRAFT-pytorch) model for text detection and the [PARSeq](https://github.com/baudm/parseq) model for text recognition. For efficient inference, both models have been converted to the ONNX format and are executed using ONNX Runtime.

The application, built with FastAPI, offers two primary endpoints: one that returns predictions as JSON and another that provides an interactive demo built with Gradio.


## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sgalle9/text-recognition-api.git
    cd text-recognition-api
    ```

2. **Add assets and models:**
    - Place a sample image inside the `assets` folder and ensure it is named `example.jpg`. This will be used as the default example in the Gradio demo.
    - Download the `.onnx` model files from the Releases page and place them inside the `models` folder.

3. **(Optional) Configure the API:** You can modify the API's global constants by editing the `app/config.py` file.

4. **Build the Docker image:**
    ```bash
    docker build -t ocr .
    ```


## Usage

1. **Run the API:**

    ```bash
    docker run --rm -it -p 7860:7860 ocr
    ```

2. **Access the Endpoints:**

    - Interactive Demo: Go to http://localhost:7860/demo.
    - API Documentation: Go to http://localhost:7860/docs.
