from __future__ import annotations

import io
from functools import lru_cache


@lru_cache(maxsize=1)
def get_ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr_image_bytes(data: bytes) -> str:
    import numpy as np
    from PIL import Image
    image = Image.open(io.BytesIO(data)).convert("RGB")
    result, _ = get_ocr_engine()(np.asarray(image))
    if not result:
        return ""
    return "\n".join(str(row[1]) for row in result if len(row) > 1 and row[1])
