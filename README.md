# 📄 Doc Cropper
A simple **document cropper + OCR web app** built with **OpenCV, FastAPI, and Tesseract/EasyOCR**.  

Upload a photo of a receipt/document → crop/warp → run OCR → download JSON + images.

---

## 🚀 Features
- 📐 Perspective transform (auto-detect document edges & crop)
- 🖼️ Image binarization (OCR-ready)
- 🔠 OCR engines:
  - Tesseract (default, lightweight)
  - EasyOCR (for noisy/complex cases)
- 🌐 FastAPI backend with clean UI
- 📥 Download scanned image & OCR JSON
- 📱 Mobile-friendly responsive design

---

## 🚀 Demo

[![Demo Video](https://img.youtube.com/vi/8nZu9voAyiM/0.jpg)](https://www.youtube.com/watch?v=8nZu9voAyiM)

---

## 📂 Project Structure

```
doc-cropper-ocr/
├── docscanner/        # Core OpenCV document detection + warp
├── web_demo/          # Web demo (FastAPI + UI)
│   ├── app.py         # FastAPI app
│   ├── templates/     # Jinja2 HTML templates
│   └── static/        # CSS / JS / assets
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignore venv, cache, etc.
└── README.md
```

---

## 🛠️ Installation

```bash
git clone git@github.com:JosephDavisC/doc-cropper-ocr.git
cd doc-cropper-ocr
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## ▶️ Run the Web App

```bash
uvicorn web_demo.app:app --reload --port 8000
```

Then open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📸 Demo

1. Upload or drop an image (receipt/document).
2. See auto-cropped & binarized result.
3. OCR extracts text + parsed fields (total, tax, date).
4. Download JSON + processed image.

---
