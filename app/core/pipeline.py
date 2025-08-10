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

import math
from typing import Any

import cv2
import numpy as np
import onnxruntime

from app.core.decoder import Decoder
from app.core.utils import softmax


class STDRPipeline:
    """Scene Text Detection and Recognition Pipeline"""

    def __init__(self, config: dict[str, Any]):

        # Text Detection
        self.detection_model = onnxruntime.InferenceSession(config["detection"]["path"])
        self.detection_input_name = self.detection_model.get_inputs()[0].name
        self.detection_mean = np.array(config["detection"]["mean"], dtype=np.float32)
        self.detection_std = np.array(config["detection"]["std"], dtype=np.float32)

        # Postprocess Detection
        postprocess_config = config["postprocess"]
        self.text_threshold = postprocess_config["text_threshold"]
        self.link_threshold = postprocess_config["link_threshold"]
        self.low_text = postprocess_config["low_text"]
        self.area_threshold = postprocess_config["area_threshold"]

        # The recognition model expects images (text patches) of this size.
        self.patch_h, self.patch_w = postprocess_config["patch_size"]
        self.dst_coords = np.array(
            [[0, 0], [self.patch_w - 1, 0], [self.patch_w - 1, self.patch_h - 1], [0, self.patch_h - 1]],
            dtype="float32",
        )

        # Text Recognition
        self.decoder = Decoder(**config["decoder"])
        self.recognition_model = onnxruntime.InferenceSession(config["recognition"]["path"])
        self.recognition_input_name = self.recognition_model.get_inputs()[0].name
        self.recognition_mean = np.array(config["recognition"]["mean"], dtype=np.float32)
        self.recognition_std = np.array(config["recognition"]["std"], dtype=np.float32)

    def _postprocess_detection(
        self,
        image: np.ndarray,
        score_map: np.ndarray,
        text_threshold: float | None = None,
        link_threshold: float | None = None,
        low_text: float | None = None,
        area_threshold: float | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Identifies text bounding boxes and extracts the corresponding patches from the original image.

        This method is a modified version of the `getDetBoxes` helper function from
        the official CRAFT repository: https://github.com/clovaai/CRAFT-pytorch.

        Original Work Copyright (c) 2019-present NAVER Corp.
        Licensed under the MIT License.
        """

        if text_threshold is None:
            text_threshold = self.text_threshold
        if link_threshold is None:
            link_threshold = self.link_threshold
        if low_text is None:
            low_text = self.low_text
        if area_threshold is None:
            area_threshold = self.area_threshold

        image_h, image_w = image.shape[:-1]
        textmap = cv2.resize(score_map[0], (image_w, image_h), interpolation=cv2.INTER_AREA)
        linkmap = cv2.resize(score_map[1], (image_w, image_h), interpolation=cv2.INTER_AREA)

        ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
        ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

        text_score_comb = np.clip(text_score + link_score, 0, 1)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_score_comb.astype(np.uint8), connectivity=4
        )

        boxes = []
        text_patches = []
        for k in range(1, nLabels):
            # ==================================================
            # Detection Box Calculation
            # ==================================================
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < area_threshold:
                continue

            # thresholding
            if np.max(textmap[labels == k]) < text_threshold:
                continue

            # make segmentation map
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels == k] = 255
            segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            # boundary check
            sx = max(x - niter, 0)
            sy = max(y - niter, 0)
            ex = min(x + w + niter + 1, image_w)
            ey = min(y + h + niter + 1, image_h)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
            rectangle = cv2.minAreaRect(np_contours)

            # Check for a degenerate rectangle (zero width or height).
            width, height = rectangle[1]
            if width < 1 or height < 1:
                continue

            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - startidx, 0)
            box = np.array(box)

            # In general, boxes containing words are wider than longer.
            # Hence, we rotate the box when width > height.
            w = max(int(np.linalg.norm(box[2] - box[3])), int(np.linalg.norm(box[1] - box[0])))
            h = max(int(np.linalg.norm(box[1] - box[2])), int(np.linalg.norm(box[0] - box[3])))
            if h > w:
                box = np.roll(box, -1, axis=0)
            boxes.append(box)

            # ==================================================
            # Image Patch Extraction
            # ==================================================
            M = cv2.getPerspectiveTransform(np.array(box, dtype="float32"), self.dst_coords)

            text_patch = cv2.warpPerspective(image, M, (self.patch_w, self.patch_h), flags=cv2.INTER_CUBIC)

            text_patches.append(text_patch)

        if text_patches:
            text_patches = np.stack(text_patches, axis=0)

        return text_patches, boxes

    def __call__(
        self,
        image,
        text_threshold: float | None = None,
        link_threshold: float | None = None,
        low_text: float | None = None,
        area_threshold: float | None = None,
    ) -> list[tuple[str, float, np.ndarray]]:
        """Runs the full detection and recognition pipeline on an input image."""

        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.float32)

        image = (image - image.min()) / (image.max() - image.min())

        # ==================================================
        # Text Detection
        # ==================================================
        detection_input = (image - self.detection_mean) / self.detection_std
        detection_input = np.moveaxis(detection_input, -1, -3)[np.newaxis, ...]
        detection_input = {self.detection_input_name: detection_input}

        score_map = self.detection_model.run(None, detection_input)[0][0]

        # Text Detection Post-Processing
        text_patches, boxes = self._postprocess_detection(
            image, score_map, text_threshold, link_threshold, low_text, area_threshold
        )

        prediction = []
        if len(text_patches) > 0:
            # ==================================================
            # Text Recognition
            # ==================================================
            recognition_input = (text_patches - self.recognition_mean) / self.recognition_std
            recognition_input = np.moveaxis(recognition_input, -1, -3)
            recognition_input = {self.recognition_input_name: recognition_input}

            logits = self.recognition_model.run(None, recognition_input)[0]

            probas = softmax(logits, axis=-1)
            labels, probas = self.decoder(probas)
            word_probas = [np.prod(p).item() for p in probas]

            prediction = list(zip(labels, word_probas, boxes))

        return prediction
