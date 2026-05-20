from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


VARIANTS = {
    "baseline": Path("outputs/test_nichtmehrkrisebilder_baseline"),
    "mild_redboost": Path("outputs/test_nichtmehrkrisebilder_mild_redboost"),
    "medium_redboost": Path("outputs/test_nichtmehrkrisebilder_medium_redboost"),
    "strong_redboost": Path("outputs/test_nichtmehrkrisebilder_strong_redboost"),
}


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def mask_bgr(mask_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def main() -> None:
    out_csv = Path("outputs/redboost_sweep_summary.csv")
    out_txt = Path("outputs/redboost_sweep_summary.txt")
    out_vis = Path("outputs/redboost_sweep_visuals")
    out_vis.mkdir(parents=True, exist_ok=True)

    baseline_root = VARIANTS["baseline"]
    rel_masks = sorted(p.relative_to(baseline_root) for p in baseline_root.glob("*/*/view_*/selected_mask.png"))
    if not rel_masks:
        raise FileNotFoundError(f"No baseline masks found under {baseline_root}")

    rows: list[dict[str, object]] = []
    per_view_best: dict[str, str] = {}

    for rel in rel_masks:
        b_mask = read_gray(baseline_root / rel) > 0
        b_depth = read_gray(baseline_root / rel.parent / "depth_masked.png") > 0
        best_variant = "baseline"
        best_area = int(b_mask.sum())

        for variant, root in VARIANTS.items():
            mask = read_gray(root / rel) > 0
            depth = read_gray(root / rel.parent / "depth_masked.png") > 0
            inter = int((b_mask & mask).sum())
            union = int((b_mask | mask).sum())
            area = int(mask.sum())
            if variant != "baseline" and area > best_area:
                best_area = area
                best_variant = variant
            rows.append(
                {
                    "sample_view": str(rel.parent),
                    "variant": variant,
                    "mask_area_px": area,
                    "area_delta_vs_baseline": area - int(b_mask.sum()),
                    "depth_nonzero_px": int(depth.sum()),
                    "depth_delta_vs_baseline": int(depth.sum()) - int(b_depth.sum()),
                    "iou_vs_baseline": 1.0 if union == 0 else inter / union,
                }
            )
        per_view_best[str(rel.parent)] = best_variant

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = ["variant;mean_area_delta;mean_depth_delta;mean_iou;views_best"]
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        vrows = [r for r in rows if r["variant"] == variant]
        summary_lines.append(
            f"{variant};"
            f"{np.mean([float(r['area_delta_vs_baseline']) for r in vrows]):.1f};"
            f"{np.mean([float(r['depth_delta_vs_baseline']) for r in vrows]):.1f};"
            f"{np.mean([float(r['iou_vs_baseline']) for r in vrows]):.4f};"
            f"{sum(1 for v in per_view_best.values() if v == variant)}"
        )
    out_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    for rel in rel_masks:
        sample_view = str(rel.parent)
        best = per_view_best[sample_view]
        overlays = [
            label(read_bgr(VARIANTS[v] / rel.parent / "overlay.png"), v)
            for v in ("baseline", "mild_redboost", "medium_redboost", "strong_redboost")
        ]
        bmask = label(mask_bgr(read_gray(VARIANTS["baseline"] / rel)), "baseline mask")
        best_mask = label(mask_bgr(read_gray(VARIANTS[best] / rel)), f"best mask: {best}")
        panel = np.vstack([np.hstack(overlays[:2]), np.hstack(overlays[2:]), np.hstack([bmask, best_mask])])
        out_name = sample_view.replace("/", "__") + ".png"
        cv2.imwrite(str(out_vis / out_name), panel)

    print(f"[DONE] wrote {out_csv}, {out_txt}, and {len(rel_masks)} panels under {out_vis}")


if __name__ == "__main__":
    main()
