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

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def resize_bbox(bbox: np.ndarray, ratio_x: float, ratio_y: float) -> np.ndarray:
    bbox.copy()
    bbox[:, 0] = bbox[:, 0] * ratio_x
    bbox[:, 1] = bbox[:, 1] * ratio_y
    return bbox


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    # Subtract the max for numerical stability.
    max_logits = np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(logits - max_logits)

    sum_exps = np.sum(exps, axis=axis, keepdims=True)
    probas = exps / sum_exps

    return probas


def plot_predictions(image, predictions: list[tuple[str, float, np.ndarray]]) -> Figure:
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")

    for label, proba, bbox in predictions:
        poly = patches.Polygon(bbox, closed=True, edgecolor="g", facecolor="none", linewidth=1.5)
        ax.add_patch(poly)

        top_edge = sorted(bbox, key=lambda p: p[1])[:2]
        top_left = min(top_edge, key=lambda p: p[0])
        top_right = max(top_edge, key=lambda p: p[0])

        tan = abs((top_left[1] - top_right[1]) / (top_left[0] - top_right[0]))
        angle = math.degrees(math.atan(tan))

        ax.text(
            x=top_left[0],
            y=top_left[1],
            s=f"{label} ({proba:.2f})",
            color="white",
            fontsize=6,
            rotation=angle,
            rotation_mode="anchor",
            horizontalalignment="left",
            verticalalignment="bottom",
            bbox={"facecolor": "green", "alpha": 0.7, "edgecolor": "none", "boxstyle": "round,pad=0.2"},
        )

    plt.tight_layout()

    return fig
