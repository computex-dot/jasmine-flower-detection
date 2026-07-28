"""
Model loading and inference logic, ported directly from the original
Streamlit app's load_model / run_inference / class_map functions.
"""

import gc
import threading
from typing import List, Tuple

import numpy as np
import torch

# Limit PyTorch CPU thread pool size to prevent high RAM allocation from OpenMP
torch.set_num_threads(1)

CLASS_MAP = {0: "Bud", 1: "Flower"}


class ModelWrapper:
    def __init__(self, weights_path: str, yolov7_repo_path: str = "yolov7"):
        self.weights_path = weights_path
        self.yolov7_repo_path = yolov7_repo_path
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        """Loads the YOLOv7 model once with thread safety and minimal memory overhead."""
        with self._lock:
            if self._model is not None:
                return
            torch.set_grad_enabled(False)
            self._model = torch.hub.load(
                self.yolov7_repo_path,
                "custom",
                self.weights_path,
                source="local",
                trust_repo=True,
            )
            self._model.eval()
            gc.collect()

    def is_loaded(self) -> bool:
        return self._model is not None

    @torch.no_grad()
    def run_inference(self, img: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """Runs detection on a single RGB image array with zero gradient overhead."""
        if self._model is None:
            self.load()

        with torch.no_grad():
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

            rendered = results.render()
            annotated_img = rendered[0]

        gc.collect()
        return detections, annotated_img
