"""
Model loading and inference logic, ported directly from the original
Streamlit app's load_model / run_inference / class_map functions.
"""

from typing import List, Tuple

import numpy as np
import torch

CLASS_MAP = {0: "Bud", 1: "Flower"}


class ModelWrapper:
    def __init__(self, weights_path: str, yolov7_repo_path: str = "yolov7"):
        self.weights_path = weights_path
        self.yolov7_repo_path = yolov7_repo_path
        self._model = None

    def load(self):
        """Loads the YOLOv7 model once. Equivalent to the Streamlit
        @st.cache_resource-decorated load_model()."""
        self._model = torch.hub.load(
            self.yolov7_repo_path,
            "custom",
            self.weights_path,
            source="local",
            trust_repo=True,
        )
        self._model.eval()

    def is_loaded(self) -> bool:
        return self._model is not None

    def run_inference(self, img: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """Runs detection on a single RGB image array (already resized to
        640x640, matching the original load_image() behaviour).

        Returns:
            detections: list of dicts with class/confidence/box coordinates
            annotated_img: numpy array of the rendered image with boxes drawn
        """
        if self._model is None:
            raise RuntimeError("Model has not been loaded yet.")

        results = self._model(img)
        df = results.pandas().xyxy[0]

        detections = []
        for _, row in df.iterrows():
            detections.append(
                {
                    "class": CLASS_MAP.get(int(row["class"]), str(row["class"])),
                    "xmin": float(row["xmin"]),
                    "ymin": float(row["ymin"]),
                    "xmax": float(row["xmax"]),
                    "ymax": float(row["ymax"]),
                    "confidence": round(float(row["confidence"]), 4),
                }
            )

        # results.render() draws boxes in-place and returns a list of arrays,
        # one per image in the batch — we only send a single image at a time.
        rendered = results.render()
        annotated_img = rendered[0]

        return detections, annotated_img
