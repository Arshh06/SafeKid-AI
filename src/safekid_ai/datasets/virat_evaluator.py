"""Rigorous detection and tracking evaluation engine for surveillance benchmarks."""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from safekid_ai.datasets.virat_annotations import VIRATObjectDetection
from safekid_ai.detector import DetectionBox
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.datasets.virat_evaluator")


def calculate_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Computes Intersection over Union (IoU) between two (x1, y1, x2, y2) bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


@dataclass
class VIRATDetectionMetrics:
    """Quantitative performance metrics for person detection on surveillance footage."""
    evaluated_frames: int = 0
    total_ground_truth_persons: int = 0
    total_predicted_persons: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ap_50: float = 0.0
    ap_50_95: float = 0.0
    mean_iou: float = 0.0
    mean_inference_latency_ms: float = 0.0
    throughput_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluated_frames": self.evaluated_frames,
            "total_ground_truth_persons": self.total_ground_truth_persons,
            "total_predicted_persons": self.total_predicted_persons,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "ap_50": round(self.ap_50, 4),
            "ap_50_95": round(self.ap_50_95, 4),
            "mean_iou": round(self.mean_iou, 4),
            "mean_inference_latency_ms": round(self.mean_inference_latency_ms, 2),
            "throughput_fps": round(self.throughput_fps, 2),
        }


@dataclass
class VIRATTrackingMetrics:
    """Quantitative performance metrics for multi-object tracking (MOT) on surveillance footage."""
    evaluated_frames: int = 0
    total_gt_tracks: int = 0
    total_predicted_tracks: int = 0
    id_switches: int = 0
    fragmentations: int = 0
    mota: float = 0.0
    idf1: float = 0.0
    id_precision: float = 0.0
    id_recall: float = 0.0
    mean_track_continuity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluated_frames": self.evaluated_frames,
            "total_gt_tracks": self.total_gt_tracks,
            "total_predicted_tracks": self.total_predicted_tracks,
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "mota": round(self.mota, 4),
            "idf1": round(self.idf1, 4),
            "id_precision": round(self.id_precision, 4),
            "id_recall": round(self.id_recall, 4),
            "mean_track_continuity": round(self.mean_track_continuity, 4),
        }


class VIRATDetectionEvaluator:
    """Evaluates detector bounding boxes against ground-truth surveillance annotations."""

    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold
        # Stores per-frame data: frame_id -> (list of GT boxes, list of (pred_box, conf))
        self.records: Dict[int, Dict[str, Any]] = {}
        self.inference_latencies_ms: List[float] = []

    def record_frame(
        self,
        frame_id: int,
        ground_truth: List[VIRATObjectDetection],
        predictions: List[DetectionBox],
        inference_time_ms: float = 0.0,
    ) -> None:
        """Records ground truth and predictions for a frame."""
        gt_boxes = [gt.bbox_xyxy for gt in ground_truth if gt.is_person]
        pred_items = [(p.bbox_xyxy, float(p.confidence)) for p in predictions]

        self.records[frame_id] = {
            "gt": gt_boxes,
            "pred": pred_items,
        }
        if inference_time_ms > 0:
            self.inference_latencies_ms.append(inference_time_ms)

    def compute_metrics(self) -> VIRATDetectionMetrics:
        """Calculates standard Precision, Recall, F1, mAP@0.50, and mAP@0.50:0.95."""
        if not self.records:
            return VIRATDetectionMetrics()

        total_gt = sum(len(r["gt"]) for r in self.records.values())
        total_pred = sum(len(r["pred"]) for r in self.records.values())

        # 1. Frame-level Greedy Matching at primary iou_threshold (0.50)
        tp_count = 0
        fp_count = 0
        matched_ious: List[float] = []

        for r in self.records.values():
            gt_boxes = list(r["gt"])
            pred_items = sorted(r["pred"], key=lambda x: x[1], reverse=True)
            matched_gt = set()

            for p_box, conf in pred_items:
                best_iou = 0.0
                best_idx = -1
                for idx, g_box in enumerate(gt_boxes):
                    if idx in matched_gt:
                        continue
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou >= self.iou_threshold and best_idx != -1:
                    tp_count += 1
                    matched_gt.add(best_idx)
                    matched_ious.append(best_iou)
                else:
                    fp_count += 1

        fn_count = max(0, total_gt - tp_count)
        precision = float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
        recall = float(tp_count / total_gt) if total_gt > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        # 2. Compute AP@0.50 and mAP@[0.50:0.95]
        ap_50 = self._compute_ap_at_iou(0.50, total_gt)
        
        # 10 IoU steps from 0.50 to 0.95
        iou_thresholds = [0.50 + 0.05 * k for k in range(10)]
        ap_steps = [self._compute_ap_at_iou(th, total_gt) for th in iou_thresholds]
        ap_50_95 = float(np.mean(ap_steps)) if ap_steps else 0.0

        mean_lat = float(np.mean(self.inference_latencies_ms)) if self.inference_latencies_ms else 0.0
        fps = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

        return VIRATDetectionMetrics(
            evaluated_frames=len(self.records),
            total_ground_truth_persons=total_gt,
            total_predicted_persons=total_pred,
            true_positives=tp_count,
            false_positives=fp_count,
            false_negatives=fn_count,
            precision=precision,
            recall=recall,
            f1_score=f1,
            ap_50=ap_50,
            ap_50_95=ap_50_95,
            mean_iou=mean_iou,
            mean_inference_latency_ms=mean_lat,
            throughput_fps=fps,
        )

    def _compute_ap_at_iou(self, iou_thresh: float, total_gt: int) -> float:
        """Computes Average Precision (AP) across all recorded frames at a specific IoU threshold."""
        if total_gt == 0:
            return 0.0

        # Collect all predictions across all frames: list of (confidence, frame_id, bbox)
        all_preds = []
        for fid, r in self.records.items():
            for p_box, conf in r["pred"]:
                all_preds.append((conf, fid, p_box))

        if not all_preds:
            return 0.0

        # Sort all predictions by confidence descending
        all_preds.sort(key=lambda x: x[0], reverse=True)

        matched_gt_per_frame: Dict[int, Set[int]] = defaultdict(set)
        tp_arr = np.zeros(len(all_preds), dtype=np.float64)
        fp_arr = np.zeros(len(all_preds), dtype=np.float64)

        for i, (conf, fid, p_box) in enumerate(all_preds):
            gt_boxes = self.records[fid]["gt"]
            matched_set = matched_gt_per_frame[fid]

            best_iou = 0.0
            best_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_set:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx

            if best_iou >= iou_thresh and best_idx != -1:
                tp_arr[i] = 1.0
                matched_set.add(best_idx)
            else:
                fp_arr[i] = 1.0

        # Cumulative precision and recall curve
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(fp_arr)
        recalls = cum_tp / total_gt
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

        # Standard 11-point interpolation or all-point AUC integration
        # VOC all-point interpolation:
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for j in range(len(mpre) - 1, 0, -1):
            mpre[j - 1] = np.maximum(mpre[j - 1], mpre[j])

        # Integrate area under curve
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        return float(np.clip(ap, 0.0, 1.0))


class VIRATTrackingEvaluator:
    """Evaluates multi-object tracking metrics (MOTA, IDF1, ID Switches, Fragmentations)."""

    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold
        # frame_id -> {gt_track_id: bbox_xyxy}
        self.gt_frames: Dict[int, Dict[int, Tuple[float, float, float, float]]] = defaultdict(dict)
        # frame_id -> {pred_track_id: bbox_xyxy}
        self.pred_frames: Dict[int, Dict[int, Tuple[float, float, float, float]]] = defaultdict(dict)

    def record_frame(
        self,
        frame_id: int,
        ground_truth: List[VIRATObjectDetection],
        tracked_predictions: List[DetectionBox],
    ) -> None:
        """Records ground truth tracks and predicted tracks for a frame."""
        for gt in ground_truth:
            if gt.is_person:
                self.gt_frames[frame_id][gt.track_id] = gt.bbox_xyxy

        for pred in tracked_predictions:
            if pred.track_id is not None:
                self.pred_frames[frame_id][pred.track_id] = pred.bbox_xyxy

    def compute_metrics(self) -> VIRATTrackingMetrics:
        """Computes MOTA, IDF1, ID Switches, and Fragmentations."""
        all_frames = sorted(set(self.gt_frames.keys()) | set(self.pred_frames.keys()))
        if not all_frames:
            return VIRATTrackingMetrics()

        all_gt_ids = {tid for f in self.gt_frames.values() for tid in f.keys()}
        all_pred_ids = {tid for f in self.pred_frames.values() for tid in f.keys()}

        total_gt_instances = sum(len(f) for f in self.gt_frames.values())
        if total_gt_instances == 0:
            return VIRATTrackingMetrics(
                evaluated_frames=len(all_frames),
                total_gt_tracks=len(all_gt_ids),
                total_predicted_tracks=len(all_pred_ids),
            )

        # 1. Frame-by-frame matching to evaluate MOTA and ID Switches
        mota_fn = 0
        mota_fp = 0
        id_switches = 0
        last_matched_pred: Dict[int, int] = {}  # gt_id -> last_pred_id

        # Co-occurrence matrix for IDF1: co_matrix[gt_id][pred_id] = count of overlapping frames
        co_matrix: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        gt_counts: Dict[int, int] = defaultdict(int)
        pred_counts: Dict[int, int] = defaultdict(int)

        # Fragmentation tracking: tracks frame gaps per GT track
        gt_presence: Dict[int, List[int]] = defaultdict(list)
        gt_detected_frames: Dict[int, List[int]] = defaultdict(list)

        for fid in all_frames:
            gt_dict = self.gt_frames.get(fid, {})
            pred_dict = self.pred_frames.get(fid, {})

            for gid in gt_dict:
                gt_counts[gid] += 1
                gt_presence[gid].append(fid)
            for pid in pred_dict:
                pred_counts[pid] += 1

            # Match GT and Preds in this frame
            matched_gts = set()
            matched_preds = set()

            # Greedy IoU matching
            pairs = []
            for gid, g_box in gt_dict.items():
                for pid, p_box in pred_dict.items():
                    iou = calculate_iou(g_box, p_box)
                    if iou >= self.iou_threshold:
                        pairs.append((iou, gid, pid))

            pairs.sort(key=lambda x: x[0], reverse=True)
            for iou, gid, pid in pairs:
                if gid not in matched_gts and pid not in matched_preds:
                    matched_gts.add(gid)
                    matched_preds.add(pid)
                    co_matrix[gid][pid] += 1
                    gt_detected_frames[gid].append(fid)

                    # Check for ID Switch
                    if gid in last_matched_pred and last_matched_pred[gid] != pid:
                        id_switches += 1
                    last_matched_pred[gid] = pid

            mota_fn += (len(gt_dict) - len(matched_gts))
            mota_fp += (len(pred_dict) - len(matched_preds))

        # MOTA calculation: 1 - (FN + FP + IDSW) / Total_GT
        mota_raw = 1.0 - float(mota_fn + mota_fp + id_switches) / float(total_gt_instances)
        mota = max(-1.0, min(1.0, mota_raw))

        # 2. IDF1 Calculation using optimal Hungarian bipartite matching (Ristani et al. 2016)
        idtp = 0
        if co_matrix:
            try:
                from scipy.optimize import linear_sum_assignment

                gt_keys = list(co_matrix.keys())
                all_pred_keys = sorted({pid for p_dict in co_matrix.values() for pid in p_dict.keys()})

                if gt_keys and all_pred_keys:
                    cost_matrix = np.zeros((len(gt_keys), len(all_pred_keys)), dtype=np.float64)
                    for i, gid in enumerate(gt_keys):
                        for j, pid in enumerate(all_pred_keys):
                            cost_matrix[i, j] = -float(co_matrix[gid].get(pid, 0))

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for r, c in zip(row_ind, col_ind):
                        idtp += co_matrix[gt_keys[r]].get(all_pred_keys[c], 0)
            except Exception as e:
                logger.debug(f"Linear sum assignment fallback to greedy: {e}")
                # Fallback to greedy matching
                matched_gt_ids: Set[int] = set()
                matched_pred_ids: Set[int] = set()
                all_co_pairs = []
                for gid, p_dict in co_matrix.items():
                    for pid, count in p_dict.items():
                        all_co_pairs.append((count, gid, pid))

                all_co_pairs.sort(key=lambda x: x[0], reverse=True)
                for count, gid, pid in all_co_pairs:
                    if gid not in matched_gt_ids and pid not in matched_pred_ids:
                        matched_gt_ids.add(gid)
                        matched_pred_ids.add(pid)
                        idtp += count

        idfp = sum(pred_counts.values()) - idtp
        idfn = sum(gt_counts.values()) - idtp

        id_prec = float(idtp / (idtp + idfp)) if (idtp + idfp) > 0 else 0.0
        id_rec = float(idtp / (idtp + idfn)) if (idtp + idfn) > 0 else 0.0
        idf1 = float(2 * idtp / (2 * idtp + idfp + idfn)) if (2 * idtp + idfp + idfn) > 0 else 0.0

        # 3. Fragmentations: count of non-consecutive detection returns
        fragmentations = 0
        continuities = []
        for gid, det_list in gt_detected_frames.items():
            if not det_list:
                continue
            total_gt_len = len(gt_presence[gid])
            continuities.append(len(det_list) / max(1, total_gt_len))

            for k in range(len(det_list) - 1):
                if det_list[k + 1] - det_list[k] > 1:
                    fragmentations += 1

        mean_continuity = float(np.mean(continuities)) if continuities else 0.0

        return VIRATTrackingMetrics(
            evaluated_frames=len(all_frames),
            total_gt_tracks=len(all_gt_ids),
            total_predicted_tracks=len(all_pred_ids),
            id_switches=id_switches,
            fragmentations=fragmentations,
            mota=mota,
            idf1=idf1,
            id_precision=id_prec,
            id_recall=id_rec,
            mean_track_continuity=mean_continuity,
        )

