from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import cv2
import numpy as np
from .utils import four_point_transform

class ProcessMode(str, Enum):
    gray = "gray"
    binary = "binary"
    contrast = "contrast"

@dataclass
class ScanResult:
    image: np.ndarray
    contour: Optional[np.ndarray]

def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Return a 4x2 float32 array of page corners (tl,tr,br,bl) or None."""
    h, w = image.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Auto Canny thresholds (sigma)
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    img_area = edges.shape[0] * edges.shape[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.015, 0.01):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                if cv2.contourArea(approx) > 0.15 * img_area:
                    quad = (approx.reshape(-1, 2).astype("float32") / scale).astype("float32")
                    return quad
    return None

def draw_contour_overlay(image: np.ndarray, contour: Optional[np.ndarray]) -> np.ndarray:
    vis = image.copy()
    if contour is not None:
        pts = contour.astype(int).reshape(-1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis

def _postprocess(img: np.ndarray, mode: ProcessMode) -> np.ndarray:
    if mode == ProcessMode.gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == ProcessMode.binary:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
    if mode == ProcessMode.contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return img

def scan_document(
    image_bgr: np.ndarray,
    mode: ProcessMode = ProcessMode.binary,
    output_size: Tuple[int, int] | None = None,
) -> ScanResult:
    contour = find_document_contour(image_bgr)
    if contour is not None:
        warped = four_point_transform(image_bgr, contour, output_size=output_size)
    else:
        warped = image_bgr.copy()
    processed = _postprocess(warped, mode)
    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    return ScanResult(processed, contour)