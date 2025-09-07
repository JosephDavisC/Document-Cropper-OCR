import numpy as np
from docscanner.core import scan_document, ProcessMode

def test_smoke():
    img = np.full((300,200,3), 255, dtype=np.uint8)
    out = scan_document(img, ProcessMode.binary).image
    assert out.shape[0] > 0