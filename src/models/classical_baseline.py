"""HOG + linear-SVM classical baseline for Experiment 4.

Object detection is recast as sliding-window classification:
extract HOG features over candidate windows, classify with a linear SVM,
and apply non-max suppression to get bounding-box predictions.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class HOGClassifier:
    """HOG-feature + linear-SVM classifier with a sliding-window detector head.

    The classifier itself is class-agnostic; the wrapper applies a sliding
    window over each test image, scores every window, and runs NMS to produce
    bounding-box predictions comparable with the YOLO baselines.
    """

    def __init__(
        self,
        win_size: tuple[int, int] = (64, 64),
        block_size: tuple[int, int] = (16, 16),
        block_stride: tuple[int, int] = (8, 8),
        cell_size: tuple[int, int] = (8, 8),
        nbins: int = 9,
        svm_C: float = 1.0,
    ) -> None:
        """Store HOG + SVM hyperparameters.

        Args:
            win_size: HOG detector window size in pixels.
            block_size: HOG block size in pixels.
            block_stride: HOG block stride in pixels.
            cell_size: HOG cell size in pixels.
            nbins: Number of orientation bins.
            svm_C: Linear-SVM regularization constant.
        """
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        self.svm_C = svm_C
        # TODO: lazily instantiate skimage.feature.hog and sklearn.svm.LinearSVC

    def extract_features(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return an (N, D) feature matrix of HOG descriptors."""
        # TODO: per-image HOG with the configured parameters; stack into matrix
        raise NotImplementedError("TODO: implement HOG feature extraction")

    def train(self, images: Sequence[np.ndarray], labels: Sequence[int]) -> None:
        """Fit the linear SVM on (image, label) pairs."""
        # TODO: extract_features; fit LinearSVC(C=self.svm_C)
        raise NotImplementedError("TODO: implement HOG+SVM training")

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run sliding-window detection on a single image.

        Returns:
            List of detections: ``[{"bbox": (x1, y1, x2, y2), "score": float, "class": int}, ...]``
        """
        # TODO: sliding window over multi-scale pyramid; score each window;
        # NMS over positive windows
        raise NotImplementedError("TODO: implement sliding-window prediction")

    def evaluate(self, images: Sequence[np.ndarray], targets: Sequence[Any]) -> dict[str, float]:
        """Compute mAP-style detection metrics against ground-truth boxes."""
        # TODO: call self.predict on each image, compare to targets, compute mAP
        raise NotImplementedError("TODO: implement evaluation")
