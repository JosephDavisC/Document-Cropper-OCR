# web_demo/app.py
from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import base64, json, re
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz
from docscanner.core import scan_document, ProcessMode

# lazy EasyOCR loader
_easyocr_reader = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="web_demo/static"), name="static")
templates = Jinja2Templates(directory="web_demo/templates")


def _to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ----------------------- Preprocess & helpers -----------------------
def _rotate_if_needed(img_bgr: np.ndarray) -> np.ndarray:
    """Use Tesseract OSD to correct orientation."""
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        osd = pytesseract.image_to_osd(rgb)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        if m:
            rot = int(m.group(1)) % 360
            if rot == 90:
                return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            if rot == 270:
                return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if rot == 180:
                return cv2.rotate(img_bgr, cv2.ROTATE_180)
    except Exception:
        pass
    return img_bgr


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Rotate → grayscale → upscale → unsharp → adaptive threshold → close."""
    img_bgr = _rotate_if_needed(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    th = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    if th.mean() < 127:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    return th


def _normalize_digits(s: str) -> str:
    return s.translate(
        str.maketrans(
            {"O": "0", "o": "0", "°": "0", "D": "0", "I": "1", "l": "1", "|": "1", "S": "5", "$": "5", "B": "8"}
        )
    )


def _pick_money_token(seq: List[str]) -> Optional[str]:
    seq = [_normalize_digits(x) for x in seq]
    cleaned = []
    for x in seq:
        y = re.sub(r"[^0-9\.,]", "", x)
        y = re.sub(r"(?<=\d),(?=\d{3}\b)", ",", y)  # keep 1,234 groupings
        cleaned.append(y)
    return max(cleaned, key=len) if cleaned else None


def _numval(s: str) -> float:
    """
    Robust numeric parse:
    - keep only digits + separators
    - if multiple dots -> treat dots as thousands (remove them)
    - strip commas
    - strip leading/trailing separators (e.g., '50.' -> '50')
    """
    s2 = re.sub(r"[^\d\.,]", "", s)      # keep digits, commas, dots
    if s2.count(".") > 1:                 # many dots -> remove all
        s2 = s2.replace(".", "")
    s2 = s2.replace(",", "")              # drop commas
    s2 = s2.strip(".,")                   # remove stray ends
    return float(s2) if re.search(r"\d", s2) else 0.0


def _looks_money(token: str) -> bool:
    # quick sanity: must include at least one digit
    if not re.search(r"\d", token):
        return False
    if len(token) > 16:                   # avoid absurdly long IDs
        return False
    v = _numval(token)
    return 100 <= v <= 1_000_000_000      # tweak if needed


def _coerce_year(date_str: str, all_text: str) -> str:
    years = re.findall(r"\b20\d{2}\b", all_text)
    target_year = None
    if years:
        from collections import Counter
        cnt = Counter(years)
        target_year = sorted(cnt.items(), key=lambda kv: (-kv[1], abs(int(kv[0]) - 2024)))[0][0]

    m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-])(20\d{2})\b", date_str)
    if m and target_year and m.group(2) != target_year:
        return m.group(1) + target_year

    m = re.search(r"\b(20\d{2})\b", date_str)
    if m:
        y = int(m.group(1))
        if y < 2000 or y > 2035:
            y_fix = int(target_year) if target_year else 2035
            return re.sub(r"\b20\d{2}\b", str(y_fix), date_str)
    return date_str


# ----------------------- Currency detection (explicit-only) -----------------------
def detect_currency(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (ISO_code, source) using only explicit tokens/symbols.
    Never guess; if unknown, return (None, None).
    """
    PATTERNS = {
        "USD": [r"\bUSD\b", r"US\$"],
        "UGX": [r"\bUGX\b", r"\bUSh\b", r"\bSHS\b", r"\bSH\.?S\.?\b"],
        "KES": [r"\bKES\b", r"\bKSH\b", r"KSh"],
        "TZS": [r"\bTZS\b", r"\bTSH\b", r"TSh"],
        "IDR": [r"\bIDR\b", r"\bRP\b"],
        "JPY": [r"\bJPY\b"],
        "EUR": [r"\bEUR\b"],
        "GBP": [r"\bGBP\b"],
        "CAD": [r"\bCAD\b"],
        "AUD": [r"\bAUD\b"],
    }
    t = text

    # keyword matches
    for code, pats in PATTERNS.items():
        for p in pats:
            if re.search(p, t, flags=re.I):
                return code, "keyword"

    # symbol matches (disambiguate when possible)
    if "€" in t:  return "EUR", "symbol"
    if "£" in t:  return "GBP", "symbol"
    if "¥" in t:  return "JPY", "symbol"
    if "$" in t:
        if re.search(r"\bCAD\b|C\$", t, flags=re.I): return "CAD", "symbol"
        if re.search(r"\bAUD\b|A\$", t, flags=re.I): return "AUD", "symbol"
        return "USD", "symbol"   # common default for bare '$'
    return None, None


# ----------------------- OCR engines -----------------------
def _ocr_tesseract(img_bgr: np.ndarray) -> Dict:
    proc = preprocess_for_ocr(img_bgr)
    data = pytesseract.image_to_data(
        proc, config="--oem 3 --psm 4 -l eng", output_type=pytesseract.Output.DICT
    )
    lines_map: Dict[tuple, List[str]] = {}
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < 60:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines_map.setdefault(key, []).append(text)
    lines = [" ".join(tokens) for _, tokens in sorted(lines_map.items()) if tokens]

    digits_txt = pytesseract.image_to_string(
        proc, config="--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=0123456789.,:/-"
    )
    digits_lines = [ln.strip() for ln in digits_txt.splitlines() if ln.strip()]
    return {"engine": "tesseract", "lines": lines, "digits_lines": digits_lines, "text": "\n".join(lines)}


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=True)  # falls back to CPU if no GPU
    return _easyocr_reader


def _ocr_easyocr(img_bgr: np.ndarray) -> Dict:
    proc = preprocess_for_ocr(img_bgr)
    reader = _get_easyocr_reader()
    lines = reader.readtext(proc, detail=0, paragraph=True)  # merged lines
    digits_lines = [ln for ln in lines if re.search(r"[0-9]", ln)]
    return {"engine": "easyocr", "lines": lines, "digits_lines": digits_lines, "text": "\n".join(lines)}


def run_ocr(img_bgr: np.ndarray, engine: str) -> Dict:
    if engine == "easyocr":
        try:
            return _ocr_easyocr(img_bgr)
        except Exception as e:
            return _ocr_tesseract(img_bgr) | {"engine": "tesseract", "note": f"easyocr failed: {e}"}
    return _ocr_tesseract(img_bgr)


# ----------------------- Parsing -----------------------
def parse_fields(ocr: Dict) -> dict:
    lines: List[str] = ocr["lines"]
    joined = " ".join(lines)

    # currency (explicit only; no guessing)
    currency, currency_source = detect_currency(joined)

    # date
    date = None
    jn = _normalize_digits(joined).replace("—", "-").replace("–", "-")
    for pat in [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*\d{2,4}\b",
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*\d{1,2},\s*\d{2,4}\b",
    ]:
        m = re.search(pat, jn, flags=re.I)
        if m:
            date = m.group(0)
            break
    if date:
        date = _coerce_year(date, joined)

    # total (smart heuristics for receipts)
    LABELS = ["TOTAL", "AMOUNT DUE", "GRAND TOTAL", "BALANCE", "CASH", "NET TOTAL"]
    BLACKLIST = [
        "TIN","T.I.N","TEL","PHONE","VERIFICATION","VERIFICATIONCODE",
        "FISCALDOC","FISCAL DOC","DOC NO","RECEIPT NO","EFRIS","QR","CODE","NIN","VATIN",
    ]

    best_idx, best_score = -1, 0
    for i, ln in enumerate(lines):
        u = ln.upper()
        score = max(fuzz.partial_ratio(u, lab) for lab in LABELS)
        if score > best_score:
            best_score, best_idx = score, i
    has_label = best_idx >= 0 and best_score >= 70

    candidates: List[tuple[str, float]] = []  # (token, score)

    def add_candidates_from_line(ln: str, idx: int, base_weight: float):
        u = ln.upper()
        if any(b in u for b in BLACKLIST):
            return
        toks = re.findall(r"[0-9\.,]{2,}", ln)
        for t in toks:
            tok = _pick_money_token([t])
            if not tok or not _looks_money(tok):
                continue
            bonus = 0.5 if "," in tok else 0.0
            score = base_weight + bonus
            candidates.append((tok, score))

    if has_label:
        for j in range(max(0, best_idx - 3), min(len(lines), best_idx + 4)):
            add_candidates_from_line(lines[j], j, base_weight=2.0 - 0.2 * abs(j - best_idx))

    start = int(len(lines) * 0.6)
    for j in range(start, len(lines)):
        add_candidates_from_line(lines[j], j, base_weight=1.0)

    digits_lines = ocr.get("digits_lines", [])
    start_d = int(len(digits_lines) * 0.6)
    for j, ln in enumerate(digits_lines):
        base = 0.8 if j >= start_d else 0.4
        add_candidates_from_line(ln, j, base)

    total = None
    if candidates:
        best_by_tok: Dict[str, float] = {}
        for tok, sc in candidates:
            best_by_tok[tok] = max(best_by_tok.get(tok, 0.0), sc)
        total = sorted(best_by_tok.keys(), key=lambda t: (best_by_tok[t], _numval(t), len(t)))[-1]

    # tax (optional)
    tax = None
    for ln in lines:
        u = ln.upper()
        if "VAT" in u or "TAX" in u:
            tok = _pick_money_token(re.findall(r"[0-9\.,]{2,}", ln))
            if tok and _looks_money(tok):
                tax = tok
                break

    return {
        "currency": currency,
        "currency_source": currency_source,  # new: "keyword" | "symbol" | None
        "date": date,
        "total": total,
        "tax": tax,
        "raw_lines": lines[:120],
    }


# ----------------------- Routes -----------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, engine: str = Query("tesseract", pattern="^(tesseract|easyocr)$")):
    return templates.TemplateResponse("index.html", {"request": request, "engine": engine})


@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("binary"),
    engine: str = Form("tesseract"),
):
    content = await file.read()
    npimg = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = scan_document(img, mode=ProcessMode(mode))
    ocr = run_ocr(result.image, engine)
    fields = parse_fields(ocr)

    blob = json.dumps(
        {"engine": ocr.get("engine"), "fields": fields, "text": ocr["text"]},
        ensure_ascii=False, indent=2
    )
    json_b64 = base64.b64encode(blob.encode("utf-8")).decode("utf-8")

    ctx = {
        "request": request,
        "engine": engine,
        "result": {
            "original_b64": _to_b64(img),
            "processed_b64": _to_b64(result.image),
            "json_b64": json_b64,
            "fields": fields,
        },
    }
    return templates.TemplateResponse("index.html", ctx)


@app.post("/api/scan")
async def api_scan(file: UploadFile = File(...), mode: str = Form("binary"), engine: str = Form("tesseract")):
    data = await file.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result = scan_document(img, mode=ProcessMode(mode))
    ocr = run_ocr(result.image, engine)
    fields = parse_fields(ocr)

    return JSONResponse(
        {
            "mode": mode,
            "engine": ocr.get("engine"),
            "processed_image_b64": _to_b64(result.image),
            "ocr": {"fields": fields, "text": ocr["text"]},
        }
    )