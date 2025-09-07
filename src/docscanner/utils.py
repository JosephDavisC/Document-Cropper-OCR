from typing import Tuple
import numpy as np
import cv2

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[s.argmin()]
    rect[2] = pts[s.argmax()]
    d = np.diff(pts, axis=1)
    rect[1] = pts[d.argmin()]
    rect[3] = pts[d.argmax()]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray,
                         output_size: Tuple[int, int] | None = None) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if output_size:
        w, h = output_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))