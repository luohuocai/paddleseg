# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
MASK_EXTS = (".png", ".bmp", ".tif", ".tiff")


@dataclass
class SegObject:
    sample_id: str
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    area: float
    contour: Optional[np.ndarray] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate defect miss/over-detection with the same sample-level "
            "logic used by deep defect batch scripts. Predictions are expected "
            "to be class-id masks, such as PaddleSeg pseudo_color_prediction/*.png."
        ))
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory of predicted class-id masks. Use pseudo_color_prediction, not added_prediction.")
    parser.add_argument(
        "--gt_dir",
        help="Directory of ground-truth class-id masks. Files are matched by stem.")
    parser.add_argument(
        "--gt_json_dir",
        help="Optional directory of LabelMe JSON labels. JSON labels are preferred over mask labels.")
    parser.add_argument(
        "--image_dir",
        help=(
            "Optional image directory defining the evaluation sample set. "
            "Images without labels are treated as OK samples with zero GT objects."))
    parser.add_argument(
        "--file_list",
        help=(
            "Optional PaddleSeg-style file list defining the evaluation sample set. "
            "The first column is used as the image path. Use with --dataset_root for relative paths."))
    parser.add_argument(
        "--dataset_root",
        default="",
        help="Root used to resolve relative paths in --file_list.")
    parser.add_argument(
        "--class_names",
        help="Optional class_names.txt. Line index is the class id.")
    parser.add_argument(
        "--output_dir",
        default="output/defect_miss_over_eval",
        help="Directory for summary JSON, details CSV, and optional debug images.")
    parser.add_argument(
        "--hit_iou_threshold",
        type=float,
        default=0.10,
        help="A GT object is hit when IoU with any prediction is at least this value.")
    parser.add_argument(
        "--min_gt_area",
        type=int,
        default=1,
        help="Ignore GT connected components smaller than this area in pixels.")
    parser.add_argument(
        "--min_pred_area",
        type=int,
        default=1,
        help="Ignore predicted connected components smaller than this area in pixels.")
    parser.add_argument(
        "--background_id",
        type=int,
        default=0,
        help="Class id treated as background and ignored.")
    parser.add_argument(
        "--require_class_match",
        action="store_true",
        help=(
            "Require class id to match before an IoU hit is accepted. "
            "Default follows the referenced script and ignores class name/id when matching."))
    parser.add_argument(
        "--allow_rgb_masks",
        action="store_true",
        help=(
            "Allow RGB/RGBA masks by reading only the first channel. "
            "Use this only when the first channel stores class ids."))
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="Save overlay images showing hit GT, missed GT, hit predictions, and over predictions.")
    return parser.parse_args()


def read_class_names(path: Optional[str]) -> List[str]:
    if not path:
        return ["_background_"]
    class_path = Path(path)
    if not class_path.exists():
        raise FileNotFoundError(f"class names file does not exist: {class_path}")
    names = [
        line.strip()
        for line in class_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return names or ["_background_"]


def class_name_from_id(class_names: Sequence[str], class_id: int) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    if class_id == 0:
        return "_background_"
    return f"class_{class_id}"


def iter_image_files(image_dir: Path) -> Iterable[Path]:
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def collect_samples_from_image_dir(image_dir: Path) -> List[Tuple[str, Optional[Path]]]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
    samples = [(path.stem, path) for path in iter_image_files(image_dir)]
    samples.sort(key=lambda item: item[0])
    return samples


def collect_samples_from_file_list(file_list: Path,
                                   dataset_root: Path) -> List[Tuple[str, Optional[Path]]]:
    if not file_list.exists():
        raise FileNotFoundError(f"file_list does not exist: {file_list}")
    samples = []
    for line in file_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        image_part = line.split()[0]
        image_path = Path(image_part)
        if not image_path.is_absolute():
            image_path = dataset_root / image_path
        samples.append((image_path.stem, image_path))
    samples.sort(key=lambda item: item[0])
    return samples


def index_files(folder: Optional[Path], exts: Sequence[str]) -> Dict[str, Path]:
    if folder is None:
        return {}
    if not folder.exists():
        raise FileNotFoundError(f"directory does not exist: {folder}")
    indexed = {}
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        indexed.setdefault(path.stem, path)
    return indexed


def collect_samples(pred_index: Dict[str, Path],
                    gt_index: Dict[str, Path],
                    json_index: Dict[str, Path],
                    args) -> Tuple[List[Tuple[str, Optional[Path]]], str]:
    if args.file_list:
        root = Path(args.dataset_root) if args.dataset_root else Path(".")
        return collect_samples_from_file_list(Path(args.file_list), root), "file_list"
    if args.image_dir:
        return collect_samples_from_image_dir(Path(args.image_dir)), "image_dir"

    stems = set(pred_index.keys()) | set(gt_index.keys()) | set(json_index.keys())
    samples = [(stem, None) for stem in sorted(stems)]
    return samples, "union_of_pred_and_gt"


def load_mask_array(path: Path, allow_rgb_masks: bool) -> np.ndarray:
    with Image.open(path) as img:
        mode = img.mode
        arr = np.array(img)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and allow_rgb_masks:
        return arr[:, :, 0]

    raise ValueError(
        f"{path} is a {mode} image with shape {arr.shape}. "
        "Please use class-id masks such as pseudo_color_prediction/*.png. "
        "If the first channel stores class ids, pass --allow_rgb_masks.")


def bbox_to_contour(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def contour_bbox(contour: np.ndarray) -> Tuple[float, float, float, float]:
    x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
    return float(x), float(y), float(x + w), float(y + h)


def objects_from_mask(path: Path,
                      class_names: Sequence[str],
                      min_area: int,
                      background_id: int,
                      allow_rgb_masks: bool,
                      sample_id: Optional[str] = None) -> List[SegObject]:
    mask = load_mask_array(path, allow_rgb_masks)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    objects = []
    sample = sample_id or path.stem
    for class_id in np.unique(mask).tolist():
        class_id = int(class_id)
        if class_id == background_id:
            continue
        class_mask = (mask == class_id).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            class_mask, connectivity=8)
        class_name = class_name_from_id(class_names, class_id)
        for component_id in range(1, num_labels):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            roi = (class_mask[y:y + h, x:x + w] > 0).astype(np.uint8)
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            contour = None
            if contours:
                best = max(contours, key=cv2.contourArea).reshape(
                    -1, 2).astype(np.float32)
                best[:, 0] += float(x)
                best[:, 1] += float(y)
                if best.shape[0] >= 3 and abs(cv2.contourArea(best)) > 0:
                    contour = best
            objects.append(
                SegObject(
                    sample_id=sample,
                    class_id=class_id,
                    class_name=class_name,
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    area=float(area),
                    contour=contour))
    return objects


def objects_from_labelme_json(path: Path,
                              class_names: Sequence[str],
                              sample_id: Optional[str] = None) -> List[SegObject]:
    data = json.loads(path.read_text(encoding="utf-8"))
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        return []

    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    objects = []
    sample = sample_id or path.stem
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        class_name = str(shape.get("label", "")).strip()
        if not class_name:
            continue
        points = shape.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        try:
            raw_points = np.array(points, dtype=np.float32).reshape(-1, 2)
        except Exception:
            continue
        if raw_points.shape[0] < 2:
            continue
        if raw_points.shape[0] >= 3:
            contour = raw_points
        else:
            x1, y1 = raw_points[0]
            x2, y2 = raw_points[1]
            contour = bbox_to_contour(
                (float(min(x1, x2)), float(min(y1, y2)),
                 float(max(x1, x2)), float(max(y1, y2))))
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        area = float(abs(cv2.contourArea(contour))) if contour.shape[
            0] >= 3 else float(w * h)
        if area <= 0:
            area = float(w * h)
        class_id = name_to_id.get(class_name, -1)
        objects.append(
            SegObject(
                sample_id=sample,
                class_id=class_id,
                class_name=class_name,
                bbox=(float(x), float(y), float(x + w), float(y + h)),
                area=area,
                contour=contour))
    return objects


def intersection_area(a: Tuple[float, float, float, float],
                      b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)


def bbox_iou(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def contour_iou(a: SegObject, b: SegObject) -> float:
    if a.contour is None or b.contour is None:
        return bbox_iou(a.bbox, b.bbox)

    ax1, ay1, ax2, ay2 = contour_bbox(a.contour)
    bx1, by1, bx2, by2 = contour_bbox(b.contour)
    x1 = int(math.floor(min(ax1, bx1)))
    y1 = int(math.floor(min(ay1, by1)))
    x2 = int(math.ceil(max(ax2, bx2)))
    y2 = int(math.ceil(max(ay2, by2)))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return 0.0

    offset = np.array([[x1, y1]], dtype=np.float32)
    contour_a = np.rint(a.contour - offset).astype(np.int32)
    contour_b = np.rint(b.contour - offset).astype(np.int32)
    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_a, [contour_a.reshape(-1, 1, 2)], 1)
    cv2.fillPoly(mask_b, [contour_b.reshape(-1, 1, 2)], 1)

    inter = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    if inter <= 0:
        return 0.0
    union = int(np.count_nonzero((mask_a > 0) | (mask_b > 0)))
    if union <= 0:
        return 0.0
    return inter / union


def class_can_match(gt: SegObject, pred: SegObject,
                    require_class_match: bool) -> bool:
    return not require_class_match or gt.class_id == pred.class_id


def evaluate_hits(preds: Sequence[SegObject],
                  gts: Sequence[SegObject],
                  threshold: float,
                  require_class_match: bool) -> Tuple[List[bool], List[bool]]:
    gt_hit = [False] * len(gts)
    pred_hit = [False] * len(preds)

    for gt_idx, gt in enumerate(gts):
        for pred in preds:
            if not class_can_match(gt, pred, require_class_match):
                continue
            if contour_iou(gt, pred) >= threshold:
                gt_hit[gt_idx] = True
                break

    for pred_idx, pred in enumerate(preds):
        for gt in gts:
            if not class_can_match(gt, pred, require_class_match):
                continue
            if contour_iou(pred, gt) >= threshold:
                pred_hit[pred_idx] = True
                break

    return gt_hit, pred_hit


def make_stats() -> Dict[str, int]:
    return {
        "sample_total": 0,
        "hit_samples": 0,
        "miss_samples": 0,
        "over_samples": 0,
        "gt_object_total": 0,
        "pred_object_total": 0,
        "hit_object_total": 0,
        "miss_object_total": 0,
        "over_object_total": 0,
        "gt_label_missing": 0,
        "pred_mask_missing": 0,
        "label_failures": 0,
        "pred_failures": 0,
    }


def update_class_stats(class_stats: Dict[str, Dict[str, int]],
                       gts: Sequence[SegObject],
                       preds: Sequence[SegObject],
                       gt_hit: Sequence[bool],
                       pred_hit: Sequence[bool]) -> None:
    for idx, gt in enumerate(gts):
        stats = class_stats.setdefault(gt.class_name, {
            "gt": 0,
            "pred": 0,
            "hit": 0,
            "miss": 0,
            "over": 0
        })
        stats["gt"] += 1
        if idx < len(gt_hit) and gt_hit[idx]:
            stats["hit"] += 1
        else:
            stats["miss"] += 1

    for idx, pred in enumerate(preds):
        stats = class_stats.setdefault(pred.class_name, {
            "gt": 0,
            "pred": 0,
            "hit": 0,
            "miss": 0,
            "over": 0
        })
        stats["pred"] += 1
        if idx >= len(pred_hit) or not pred_hit[idx]:
            stats["over"] += 1


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def object_contour(obj: SegObject) -> np.ndarray:
    contour = obj.contour if obj.contour is not None else bbox_to_contour(obj.bbox)
    return np.rint(contour).astype(np.int32).reshape(-1, 1, 2)


def draw_object(image: np.ndarray, obj: SegObject, color: Tuple[int, int, int],
                text: str, thickness: int) -> None:
    contour = object_contour(obj)
    cv2.polylines(image, [contour], True, color, thickness, cv2.LINE_AA)
    pts = contour.reshape(-1, 2)
    x = max(0, int(np.min(pts[:, 0])))
    y = max(16, int(np.min(pts[:, 1])) - 6)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv2.LINE_AA)


def read_image_for_debug(image_path: Optional[Path],
                         fallback_mask_path: Optional[Path]) -> Optional[np.ndarray]:
    if image_path is not None and image_path.exists():
        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    if fallback_mask_path is None or not fallback_mask_path.exists():
        return None
    try:
        mask = load_mask_array(fallback_mask_path, allow_rgb_masks=True)
    except Exception:
        return None
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def save_debug_image(output_dir: Path,
                     sample_id: str,
                     image_path: Optional[Path],
                     pred_path: Optional[Path],
                     gts: Sequence[SegObject],
                     preds: Sequence[SegObject],
                     gt_hit: Sequence[bool],
                     pred_hit: Sequence[bool],
                     miss_sample: int,
                     over_sample: int) -> str:
    image = read_image_for_debug(image_path, pred_path)
    if image is None:
        return ""
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()

    for idx, gt in enumerate(gts):
        hit = idx < len(gt_hit) and gt_hit[idx]
        color = (0, 180, 0) if hit else (0, 0, 255)
        label = f"GT:{gt.class_name}" if hit else f"MISS:{gt.class_name}"
        draw_object(overlay, gt, color, label, 3 if hit else 5)

    for idx, pred in enumerate(preds):
        hit = idx < len(pred_hit) and pred_hit[idx]
        color = (255, 255, 0) if hit else (255, 0, 255)
        label = f"P:{pred.class_name}" if hit else f"OVER:{pred.class_name}"
        draw_object(overlay, pred, color, label, 2 if hit else 4)

    header = (
        f"GT={len(gts)} Pred={len(preds)} miss_sample={miss_sample} "
        f"over_sample={over_sample}")
    cv2.rectangle(overlay, (0, 0), (min(overlay.shape[1], 1700), 58),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(overlay, header, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

    out_path = debug_dir / f"{sample_id}__miss{miss_sample}_over{over_sample}.png"
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        return ""
    encoded.tofile(str(out_path))
    return str(out_path)


def print_summary(stats: Dict[str, int], threshold: float) -> None:
    sample_total = stats["sample_total"]
    gt_total = stats["gt_object_total"]
    pred_total = stats["pred_object_total"]
    print("\n" + "=" * 96)
    print(f"Sample-level miss/over eval (contour IoU >= {threshold:.2f})")
    print("=" * 96)
    print(
        f"{'Samples':>10}{'GTObj':>10}{'PredObj':>10}{'HitS':>10}{'MissS':>10}"
        f"{'OverS':>10}{'HitRate':>12}{'MissRate':>12}{'OverRate':>12}")
    print("-" * 96)
    print(
        f"{sample_total:>10}{gt_total:>10}{pred_total:>10}"
        f"{stats['hit_samples']:>10}{stats['miss_samples']:>10}{stats['over_samples']:>10}"
        f"{safe_rate(stats['hit_samples'], sample_total):>12.3f}"
        f"{safe_rate(stats['miss_samples'], sample_total):>12.3f}"
        f"{safe_rate(stats['over_samples'], sample_total):>12.3f}")
    print("=" * 96)
    print(
        "Object-level: "
        f"hit={stats['hit_object_total']}, miss={stats['miss_object_total']}, "
        f"over={stats['over_object_total']}, "
        f"miss_rate={safe_rate(stats['miss_object_total'], gt_total):.3f}, "
        f"over_rate={safe_rate(stats['over_object_total'], pred_total):.3f}")


def write_reports(output_dir: Path,
                  config: Dict[str, object],
                  stats: Dict[str, int],
                  class_stats: Dict[str, Dict[str, int]],
                  rows: List[Dict[str, object]]) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": config,
        "sample_level": {
            "sample_total": stats["sample_total"],
            "hit_samples": stats["hit_samples"],
            "miss_samples": stats["miss_samples"],
            "over_samples": stats["over_samples"],
            "hit_rate": safe_rate(stats["hit_samples"], stats["sample_total"]),
            "miss_rate": safe_rate(stats["miss_samples"], stats["sample_total"]),
            "over_rate": safe_rate(stats["over_samples"], stats["sample_total"]),
        },
        "object_level": {
            "gt_object_total": stats["gt_object_total"],
            "pred_object_total": stats["pred_object_total"],
            "hit_object_total": stats["hit_object_total"],
            "miss_object_total": stats["miss_object_total"],
            "over_object_total": stats["over_object_total"],
            "hit_rate": safe_rate(stats["hit_object_total"],
                                  stats["gt_object_total"]),
            "miss_rate": safe_rate(stats["miss_object_total"],
                                   stats["gt_object_total"]),
            "over_rate": safe_rate(stats["over_object_total"],
                                   stats["pred_object_total"]),
        },
        "load_stats": {
            "gt_label_missing": stats["gt_label_missing"],
            "pred_mask_missing": stats["pred_mask_missing"],
            "label_failures": stats["label_failures"],
            "pred_failures": stats["pred_failures"],
        },
        "class_stats": class_stats,
        "per_image_count": len(rows),
    }

    summary_path = output_dir / "defect_miss_over_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_path = output_dir / "defect_miss_over_details.csv"
    fieldnames = [
        "sample_id",
        "image_path",
        "gt_path",
        "pred_path",
        "gt_object_count",
        "pred_object_count",
        "hit_object_count",
        "miss_object_count",
        "over_object_count",
        "hit_sample",
        "miss_sample",
        "over_sample",
        "debug_image",
    ]
    with detail_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return summary_path, detail_path


def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    gt_json_dir = Path(args.gt_json_dir) if args.gt_json_dir else None
    output_dir = Path(args.output_dir)
    class_names = read_class_names(args.class_names)

    pred_index = index_files(pred_dir, MASK_EXTS)
    gt_index = index_files(gt_dir, MASK_EXTS)
    json_index = index_files(gt_json_dir, (".json", ))
    samples, sample_source = collect_samples(pred_index, gt_index, json_index,
                                             args)
    if not samples:
        raise RuntimeError("No samples found for evaluation.")

    stats = make_stats()
    class_stats: Dict[str, Dict[str, int]] = {}
    rows: List[Dict[str, object]] = []

    for sample_id, image_path in samples:
        pred_path = pred_index.get(sample_id)
        gt_json_path = json_index.get(sample_id)
        gt_path = gt_json_path or gt_index.get(sample_id)

        if pred_path is None:
            stats["pred_mask_missing"] += 1
            preds: List[SegObject] = []
        else:
            try:
                preds = objects_from_mask(
                    pred_path,
                    class_names,
                    min_area=args.min_pred_area,
                    background_id=args.background_id,
                    allow_rgb_masks=args.allow_rgb_masks,
                    sample_id=sample_id)
            except Exception as exc:
                stats["pred_failures"] += 1
                print(f"[warning] failed to read prediction {pred_path}: {exc}")
                preds = []

        if gt_path is None:
            stats["gt_label_missing"] += 1
            gts: List[SegObject] = []
        else:
            try:
                if gt_path.suffix.lower() == ".json":
                    gts = objects_from_labelme_json(
                        gt_path, class_names, sample_id=sample_id)
                else:
                    gts = objects_from_mask(
                        gt_path,
                        class_names,
                        min_area=args.min_gt_area,
                        background_id=args.background_id,
                        allow_rgb_masks=args.allow_rgb_masks,
                        sample_id=sample_id)
            except Exception as exc:
                stats["label_failures"] += 1
                print(f"[warning] failed to read label {gt_path}: {exc}")
                gts = []

        gt_hit, pred_hit = evaluate_hits(preds, gts, args.hit_iou_threshold,
                                         args.require_class_match)

        hit_objects = sum(1 for hit in gt_hit if hit)
        miss_objects = sum(1 for hit in gt_hit if not hit)
        over_objects = sum(1 for hit in pred_hit if not hit)

        has_gt = len(gts) > 0
        has_hit_gt = hit_objects > 0
        miss_sample = 1 if has_gt and not has_hit_gt else 0
        over_sample = 1 if (
            (not has_gt and len(preds) > 0) or
            (has_gt and not has_hit_gt and over_objects > 0)) else 0
        hit_sample = 1 if has_hit_gt or (not has_gt and len(preds) == 0) else 0

        stats["sample_total"] += 1
        stats["hit_samples"] += hit_sample
        stats["miss_samples"] += miss_sample
        stats["over_samples"] += over_sample
        stats["gt_object_total"] += len(gts)
        stats["pred_object_total"] += len(preds)
        stats["hit_object_total"] += hit_objects
        stats["miss_object_total"] += miss_objects
        stats["over_object_total"] += over_objects

        update_class_stats(class_stats, gts, preds, gt_hit, pred_hit)

        debug_image = ""
        if args.save_debug:
            debug_image = save_debug_image(
                output_dir=output_dir,
                sample_id=sample_id,
                image_path=image_path,
                pred_path=pred_path,
                gts=gts,
                preds=preds,
                gt_hit=gt_hit,
                pred_hit=pred_hit,
                miss_sample=miss_sample,
                over_sample=over_sample)

        rows.append({
            "sample_id": sample_id,
            "image_path": str(image_path) if image_path else "",
            "gt_path": str(gt_path) if gt_path else "",
            "pred_path": str(pred_path) if pred_path else "",
            "gt_object_count": len(gts),
            "pred_object_count": len(preds),
            "hit_object_count": hit_objects,
            "miss_object_count": miss_objects,
            "over_object_count": over_objects,
            "hit_sample": hit_sample,
            "miss_sample": miss_sample,
            "over_sample": over_sample,
            "debug_image": debug_image,
        })

    config = {
        "sample_source": sample_source,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir) if gt_dir else "",
        "gt_json_dir": str(gt_json_dir) if gt_json_dir else "",
        "image_dir": args.image_dir or "",
        "file_list": args.file_list or "",
        "dataset_root": args.dataset_root,
        "class_names": args.class_names or "",
        "hit_iou_threshold": args.hit_iou_threshold,
        "min_gt_area": args.min_gt_area,
        "min_pred_area": args.min_pred_area,
        "background_id": args.background_id,
        "require_class_match": args.require_class_match,
        "match_rule": (
            "GT is hit if any prediction has contour IoU >= threshold; "
            "bbox IoU is used only when contour is unavailable."),
        "sample_miss_rule":
        "A positive sample is a miss sample only when none of its GT objects is hit.",
        "sample_over_rule": (
            "An OK sample is over-detection if it has any predicted object; "
            "a positive sample with any hit GT is not counted as over-detection."),
        "class_match_rule": (
            "class id must match" if args.require_class_match else
            "class id ignored, matching follows geometry only"),
    }

    print_summary(stats, args.hit_iou_threshold)
    summary_path, detail_path = write_reports(output_dir, config, stats,
                                              class_stats, rows)
    print(f"\nSummary: {summary_path}")
    print(f"Details: {detail_path}")
    if sample_source == "union_of_pred_and_gt":
        print(
            "\nNote: sample set was inferred from pred/gt files. For true image-level "
            "OK counting, pass --image_dir or --file_list so OK images without labels "
            "and without predictions are included in the denominator.")


if __name__ == "__main__":
    main()
