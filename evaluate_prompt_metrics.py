#!/usr/bin/env python3
"""
Prompt-Compliance Metrics Evaluator for Tracking Benchmarks

Computes:
- Semantic Precision (SP): Fraction of predicted boxes matching prompt-valid GT (IoU >= threshold)
- Semantic Recall (SR): Fraction of prompt-valid GT boxes matched by predictions (IoU >= threshold)
- Prompt Coverage Ratio (PCR): Fraction of frames where the prompt-valid target is correctly tracked
- Semantic ID Switches (SID): Count of track switches from prompt-valid to prompt-invalid GT
- Distractor Confusion Rate (DCR): Fraction of predictions matched to prompt-invalid GT

Matching uses IoU >= threshold with Hungarian assignment.

Input:
- gt.json: Ground truth from CARLA dataset generator
- predictions.txt: MOT format (frame,track_id,x,y,w,h,score,-1,-1,-1)

The "prompt-valid" set is configurable:
- Default: is_target == True (single red sedan)
- Extended: color == "180,30,30" AND type_id in sedan list (all red sedans)
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# Sedan type IDs for extended prompt matching
SEDAN_TYPE_IDS = {
    "vehicle.tesla.model3",
    "vehicle.audi.a2",
    "vehicle.bmw.grandtourer",
    "vehicle.mercedes.coupe",
    "vehicle.toyota.prius",
    "vehicle.ford.mustang",
}


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1: [x1, y1, x2, y2] format (xyxy)
        box2: [x1, y1, x2, y2] format (xyxy)

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def xywh_to_xyxy(box: List[float]) -> List[int]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [int(x), int(y), int(x + w), int(y + h)]


def load_ground_truth(gt_path: str) -> dict:
    """Load ground truth JSON file."""
    with open(gt_path, 'r') as f:
        return json.load(f)


def load_predictions_mot(pred_path: str) -> Dict[int, List[dict]]:
    """
    Load predictions in MOT format.

    Format: frame,track_id,x,y,w,h,score,-1,-1,-1

    Returns:
        Dict mapping frame_id -> list of predictions
    """
    predictions = defaultdict(list)

    with open(pred_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) < 7:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6]) if parts[6] != '-1' else 1.0

            predictions[frame_id].append({
                "track_id": track_id,
                "bbox_xywh": [x, y, w, h],
                "bbox_xyxy": xywh_to_xyxy([x, y, w, h]),
                "score": score
            })

    return predictions


def is_prompt_valid(annotation: dict, mode: str = "single_target") -> bool:
    """
    Check if a GT annotation matches the prompt criteria.

    Args:
        annotation: GT annotation dict
        mode: "single_target" (is_target only) or "all_red_sedans"

    Returns:
        True if annotation is prompt-valid
    """
    if mode == "single_target":
        return annotation.get("is_target", False)

    elif mode == "all_red_sedans":
        # Check if color is red AND type is a sedan
        color = annotation.get("color", "")
        type_id = annotation.get("type_id", "")
        return color == "180,30,30" and type_id in SEDAN_TYPE_IDS

    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_gt_by_frame(gt_data: dict) -> Dict[int, List[dict]]:
    """Build dict mapping frame_id -> list of GT annotations."""
    gt_by_frame = defaultdict(list)
    for ann in gt_data["annotations"]:
        gt_by_frame[ann["image_id"]].append(ann)
    return gt_by_frame


def match_predictions_to_gt(
    predictions: List[dict],
    gt_annotations: List[dict],
    iou_threshold: float = 0.5
) -> List[Tuple[dict, Optional[dict], float]]:
    """
    Match predictions to GT using Hungarian (optimal) matching on IoU.

    Builds an IoU cost matrix and uses linear_sum_assignment to find the
    globally optimal one-to-one assignment, avoiding the mismatch artifacts
    that greedy matching can produce in multi-target scenarios.

    Returns:
        List of (prediction, matched_gt_or_None, iou)
    """
    if not predictions or not gt_annotations:
        return [(pred, None, 0.0) for pred in predictions]

    n_pred = len(predictions)
    n_gt = len(gt_annotations)

    # Build IoU matrix (n_pred x n_gt)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(gt_annotations):
            iou_matrix[i, j] = compute_iou(pred["bbox_xyxy"], gt["bbox_xyxy"])

    # Hungarian algorithm minimises cost, so use (1 - IoU) as cost
    cost_matrix = 1.0 - iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Record assigned pairs that meet the IoU threshold
    pred_to_gt: Dict[int, Tuple[int, float]] = {}
    for pi, gi in zip(pred_indices, gt_indices):
        iou = iou_matrix[pi, gi]
        if iou >= iou_threshold:
            pred_to_gt[pi] = (gi, iou)

    # Build result list covering every prediction
    matches = []
    for i, pred in enumerate(predictions):
        if i in pred_to_gt:
            gi, iou = pred_to_gt[i]
            matches.append((pred, gt_annotations[gi], iou))
        else:
            # Best IoU for reporting, even though unmatched
            best_iou = float(iou_matrix[i].max()) if n_gt > 0 else 0.0
            matches.append((pred, None, best_iou))

    return matches


def compute_metrics(
    gt_data: dict,
    predictions: Dict[int, List[dict]],
    iou_threshold: float = 0.5,
    mode: str = "single_target"
) -> Tuple[float, float, float, float, dict]:
    """
    Compute Semantic Precision, Semantic Recall, Prompt Coverage Ratio,
    and Distractor Confusion Rate.

    SP:  fraction of predicted boxes that match any prompt-valid GT
    SR:  fraction of prompt-valid GT boxes matched by any prediction
    PCR: tracking reliability over time
         single target: frames where target matched / frames where target visible
         multi-target: total matched valid GT / total visible valid GT across frames
    DCR: fraction of predictions that matched a distractor GT

    Args:
        gt_data: Ground truth data
        predictions: Predictions by frame
        iou_threshold: IoU threshold for matching
        mode: Prompt validity mode

    Returns:
        (sp, sr, pcr, dcr, detailed_stats)
    """
    gt_by_frame = build_gt_by_frame(gt_data)

    total_predictions = 0
    predictions_matching_valid = 0
    predictions_matching_distractor = 0
    total_valid_gt = 0
    valid_gt_matched = 0

    # PCR accumulators
    frames_with_valid_gt = 0
    frames_where_valid_matched = 0

    # For detailed analysis
    frame_stats = []

    all_frames = set(gt_by_frame.keys()) | set(predictions.keys())

    for frame_id in sorted(all_frames):
        frame_gt = gt_by_frame.get(frame_id, [])
        frame_preds = predictions.get(frame_id, [])

        # Identify prompt-valid GT in this frame
        valid_gt = [gt for gt in frame_gt if is_prompt_valid(gt, mode)]

        total_valid_gt += len(valid_gt)
        total_predictions += len(frame_preds)

        # Match predictions to ALL GT
        matches = match_predictions_to_gt(frame_preds, frame_gt, iou_threshold)

        frame_preds_matching_valid = 0
        frame_preds_matching_distractor = 0
        matched_valid_gt_ids = set()

        for _, matched_gt, _ in matches:
            if matched_gt is not None:
                if is_prompt_valid(matched_gt, mode):
                    frame_preds_matching_valid += 1
                    matched_valid_gt_ids.add(matched_gt["gt_id"])
                else:
                    frame_preds_matching_distractor += 1

        frame_valid_matched = len(matched_valid_gt_ids)

        # PCR: track per-frame coverage
        if len(valid_gt) > 0:
            frames_with_valid_gt += 1
            if frame_valid_matched > 0:
                frames_where_valid_matched += 1

        predictions_matching_valid += frame_preds_matching_valid
        predictions_matching_distractor += frame_preds_matching_distractor
        valid_gt_matched += frame_valid_matched

        frame_stats.append({
            "frame_id": frame_id,
            "num_predictions": len(frame_preds),
            "num_valid_gt": len(valid_gt),
            "preds_matching_valid": frame_preds_matching_valid,
            "preds_matching_distractor": frame_preds_matching_distractor,
            "valid_gt_matched": frame_valid_matched
        })

    # Compute metrics
    sp = predictions_matching_valid / total_predictions if total_predictions > 0 else 0.0
    sr = valid_gt_matched / total_valid_gt if total_valid_gt > 0 else 0.0
    dcr = predictions_matching_distractor / total_predictions if total_predictions > 0 else 0.0

    # PCR: single-target uses binary per-frame, multi-target uses matched/visible ratio
    if mode == "single_target":
        pcr = frames_where_valid_matched / frames_with_valid_gt if frames_with_valid_gt > 0 else 0.0
    else:
        pcr = valid_gt_matched / total_valid_gt if total_valid_gt > 0 else 0.0

    stats = {
        "total_predictions": total_predictions,
        "predictions_matching_valid": predictions_matching_valid,
        "predictions_matching_distractor": predictions_matching_distractor,
        "total_valid_gt": total_valid_gt,
        "valid_gt_matched": valid_gt_matched,
        "frames_with_valid_gt": frames_with_valid_gt,
        "frames_where_valid_matched": frames_where_valid_matched,
        "frame_stats": frame_stats
    }

    return sp, sr, pcr, dcr, stats


def compute_semantic_id_switches(
    gt_data: dict,
    predictions: Dict[int, List[dict]],
    iou_threshold: float = 0.5,
    mode: str = "single_target"
) -> Tuple[int, List[dict]]:
    """
    Compute Semantic ID Switches (SID).

    SID counts when a predicted track switches from matching a prompt-valid GT
    to a prompt-invalid GT (or vice versa).

    This indicates the tracker is confusing the target with distractors.

    Args:
        gt_data: Ground truth data
        predictions: Predictions by frame
        iou_threshold: IoU threshold for matching
        mode: Prompt validity mode

    Returns:
        (num_switches, switch_events)
    """
    gt_by_frame = build_gt_by_frame(gt_data)

    # Track the last matched GT validity for each track_id
    track_last_valid: Dict[int, Optional[bool]] = {}
    track_last_gt_id: Dict[int, Optional[int]] = {}

    switch_events = []
    total_switches = 0

    sorted_frames = sorted(set(gt_by_frame.keys()) | set(predictions.keys()))

    for frame_id in sorted_frames:
        frame_gt = gt_by_frame.get(frame_id, [])
        frame_preds = predictions.get(frame_id, [])

        if not frame_preds:
            continue

        # Match predictions to GT
        matches = match_predictions_to_gt(frame_preds, frame_gt, iou_threshold)

        for pred, matched_gt, iou in matches:
            track_id = pred["track_id"]

            if matched_gt is None:
                # No match - don't update (track might be lost temporarily)
                continue

            current_valid = is_prompt_valid(matched_gt, mode)
            current_gt_id = matched_gt["gt_id"]

            # Check for semantic switch
            if track_id in track_last_valid:
                last_valid = track_last_valid[track_id]
                last_gt_id = track_last_gt_id[track_id]

                # Switch occurs when validity changes (valid -> invalid or invalid -> valid)
                if last_valid is not None and current_valid != last_valid:
                    total_switches += 1
                    switch_events.append({
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "from_gt_id": last_gt_id,
                        "to_gt_id": current_gt_id,
                        "from_valid": last_valid,
                        "to_valid": current_valid
                    })

            # Update tracking state
            track_last_valid[track_id] = current_valid
            track_last_gt_id[track_id] = current_gt_id

    return total_switches, switch_events


def main():
    parser = argparse.ArgumentParser(description="Evaluate Prompt-Compliance Metrics")
    parser.add_argument("--gt", required=True, help="Path to gt.json")
    parser.add_argument("--pred", required=True, help="Path to predictions file (MOT format)")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--mode", choices=["single_target", "all_red_sedans"],
                        default="single_target", help="Prompt validity mode")
    parser.add_argument("--output", help="Optional output JSON path for detailed results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-frame stats")
    args = parser.parse_args()

    # Load data
    print(f"[INFO] Loading ground truth from: {args.gt}")
    gt_data = load_ground_truth(args.gt)

    print(f"[INFO] Loading predictions from: {args.pred}")
    predictions = load_predictions_mot(args.pred)

    print(f"[INFO] Mode: {args.mode}, IoU threshold: {args.iou_threshold}")
    print()

    # Compute metrics
    sp, sr, pcr, dcr, stats = compute_metrics(
        gt_data, predictions, args.iou_threshold, args.mode
    )

    sid_count, sid_events = compute_semantic_id_switches(
        gt_data, predictions, args.iou_threshold, args.mode
    )

    # Print results
    print("=" * 60)
    print("PROMPT-COMPLIANCE METRICS")
    print("=" * 60)
    print(f"Prompt: \"{gt_data['meta']['prompt']}\"")
    print(f"Mode: {args.mode}")
    print()
    print(f"Semantic Precision (SP): {sp:.4f}")
    print(f"  - {stats['predictions_matching_valid']} / {stats['total_predictions']} "
          f"predictions match prompt-valid GT")
    print()
    print(f"Semantic Recall (SR): {sr:.4f}")
    print(f"  - {stats['valid_gt_matched']} / {stats['total_valid_gt']} "
          f"prompt-valid GT matched by predictions")
    print()
    print(f"Prompt Coverage Ratio (PCR): {pcr:.4f}")
    print(f"  - {stats['frames_where_valid_matched']} / {stats['frames_with_valid_gt']} "
          f"frames with valid target correctly tracked")
    print()
    print(f"Semantic ID Switches (SID): {sid_count}")
    if sid_events:
        print("  Switch events:")
        for event in sid_events[:10]:  # Show first 10
            direction = "valid->invalid" if event["from_valid"] else "invalid->valid"
            print(f"    Frame {event['frame_id']}: Track {event['track_id']} "
                  f"({direction}, GT {event['from_gt_id']} -> {event['to_gt_id']})")
        if len(sid_events) > 10:
            print(f"    ... and {len(sid_events) - 10} more")
    print()
    print(f"Distractor Confusion Rate (DCR): {dcr:.4f}")
    print(f"  - {stats['predictions_matching_distractor']} / {stats['total_predictions']} "
          f"predictions matched a distractor instead of the target")
    print("=" * 60)

    # Verbose output
    if args.verbose:
        print("\nPer-frame statistics:")
        for fs in stats["frame_stats"]:
            print(f"  Frame {fs['frame_id']}: "
                  f"preds={fs['num_predictions']}, "
                  f"valid_gt={fs['num_valid_gt']}, "
                  f"preds_match_valid={fs['preds_matching_valid']}, "
                  f"valid_matched={fs['valid_gt_matched']}")

    # Save detailed results
    if args.output:
        results = {
            "meta": {
                "gt_file": args.gt,
                "pred_file": args.pred,
                "iou_threshold": args.iou_threshold,
                "mode": args.mode,
                "prompt": gt_data["meta"]["prompt"]
            },
            "metrics": {
                "semantic_precision": sp,
                "semantic_recall": sr,
                "prompt_coverage_ratio": pcr,
                "distractor_confusion_rate": dcr,
                "semantic_id_switches": sid_count
            },
            "details": {
                "stats": stats,
                "switch_events": sid_events
            }
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
