import argparse, os, glob, cv2
from .core import scan_document, ProcessMode, find_document_contour, draw_contour_overlay

def process_one(path: str, out: str, mode: str, debug_dir: str | None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")

    # optional debug overlay of detected quad
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cnt = find_document_contour(img)
        overlay = draw_contour_overlay(img, cnt)
        name = os.path.basename(path).rsplit(".", 1)[0] + "_overlay.jpg"
        cv2.imwrite(os.path.join(debug_dir, name), overlay)

    res = scan_document(img, ProcessMode(mode))
    cv2.imwrite(out, res.image)

def main():
    p = argparse.ArgumentParser("Document Scanner CLI")
    p.add_argument("input")
    p.add_argument("--out", help="Output file path (single file mode)")
    p.add_argument("--out_dir", help="Output directory (folder mode)")
    p.add_argument("--glob", default="*.jpg", help="Glob for folder mode, e.g. '*.png'")
    p.add_argument("--mode", default="binary",
                   choices=[m.value for m in ProcessMode])
    p.add_argument("--debug_dir", help="Save contour overlay images here", default=None)
    a = p.parse_args()

    if os.path.isdir(a.input):
        assert a.out_dir, "--out_dir required for folder input"
        os.makedirs(a.out_dir, exist_ok=True)
        for fp in glob.glob(os.path.join(a.input, a.glob)):
            name = os.path.splitext(os.path.basename(fp))[0] + "_scan.jpg"
            process_one(fp, os.path.join(a.out_dir, name), a.mode, a.debug_dir)
            print("Saved", name)
    else:
        assert a.out, "--out required for single file"
        process_one(a.input, a.out, a.mode, a.debug_dir)
        print("Saved", a.out)

if __name__ == "__main__":
    main()