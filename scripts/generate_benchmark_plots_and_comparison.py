"""Generates benchmark comparison JSON and visualization plots for Prompt 17."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison():
    out_dir = Path("outputs")
    base_file = out_dir / "benchmark_baseline.json"
    opt_file = out_dir / "benchmark_optimized.json"

    with open(base_file, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(opt_file, "r", encoding="utf-8") as f:
        optimized_data = json.load(f)

    comparison_records = []
    scenarios = ["Swimming Pool", "Playground", "Driveway", "Boom Barrier"]

    for b, o in zip(baseline_data, optimized_data):
        sc_name = b["name"]
        res = b["video_resolution"]
        b_fps = b["average_throughput_fps"]
        o_fps = o["average_throughput_fps"]
        fps_delta = o_fps - b_fps
        fps_pct = (fps_delta / b_fps) * 100.0

        b_tot = b["stage_metrics"]["total_frame_ms"]["mean"]
        o_tot = o["stage_metrics"]["total_frame_ms"]["mean"]
        tot_delta = o_tot - b_tot
        tot_pct = (tot_delta / b_tot) * 100.0

        b_yolo = b["stage_metrics"]["yolo_inference_ms"]["mean"]
        o_yolo = o["stage_metrics"]["yolo_inference_ms"]["mean"]
        yolo_delta = o_yolo - b_yolo
        yolo_pct = (yolo_delta / b_yolo) * 100.0

        b_vis = b["stage_metrics"]["vis_and_writing_ms"]["mean"]
        o_vis = o["stage_metrics"]["vis_and_writing_ms"]["mean"]
        vis_delta = o_vis - b_vis
        vis_pct = (vis_delta / b_vis) * 100.0

        b_mem = b["peak_memory_mb"]
        o_mem = o["peak_memory_mb"]

        rec = {
            "scenario": sc_name,
            "resolution": res,
            "baseline": {
                "fps": b_fps,
                "total_latency_mean_ms": round(b_tot, 2),
                "total_latency_median_ms": round(b["stage_metrics"]["total_frame_ms"]["median"], 2),
                "total_latency_p95_ms": round(b["stage_metrics"]["total_frame_ms"]["p95"], 2),
                "yolo_latency_mean_ms": round(b_yolo, 2),
                "vis_and_writing_mean_ms": round(b_vis, 2),
                "peak_memory_mb": b_mem,
            },
            "optimized": {
                "fps": o_fps,
                "total_latency_mean_ms": round(o_tot, 2),
                "total_latency_median_ms": round(o["stage_metrics"]["total_frame_ms"]["median"], 2),
                "total_latency_p95_ms": round(o["stage_metrics"]["total_frame_ms"]["p95"], 2),
                "yolo_latency_mean_ms": round(o_yolo, 2),
                "vis_and_writing_mean_ms": round(o_vis, 2),
                "peak_memory_mb": o_mem,
            },
            "improvements": {
                "fps_gain": round(fps_delta, 2),
                "fps_gain_pct": round(fps_pct, 1),
                "total_latency_reduction_ms": round(-tot_delta, 2),
                "total_latency_reduction_pct": round(-tot_pct, 1),
                "yolo_latency_reduction_ms": round(-yolo_delta, 2),
                "yolo_latency_reduction_pct": round(-yolo_pct, 1),
                "vis_latency_reduction_ms": round(-vis_delta, 2),
                "vis_latency_reduction_pct": round(-vis_pct, 1),
            }
        }
        comparison_records.append(rec)

    # Save comparison JSON
    comp_json_path = out_dir / "prompt_17_benchmark_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)
    print(f"[Saved Comparison JSON] {comp_json_path}")

    # Generate Plots
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Plot 1: FPS Comparison
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    x = np.arange(len(scenarios))
    width = 0.35

    base_fps = [r["baseline"]["fps"] for r in comparison_records]
    opt_fps = [r["optimized"]["fps"] for r in comparison_records]

    rects1 = ax.bar(x - width/2, base_fps, width, label="Baseline (Unoptimized)", color="#94a3b8", edgecolor="#64748b")
    rects2 = ax.bar(x + width/2, opt_fps, width, label="Optimized (P-Core, Fused, Sub-ROI Vis)", color="#0ea5e9", edgecolor="#0284c7")

    ax.set_ylabel("Processing Throughput (FPS)", fontsize=12, fontweight="bold")
    ax.set_title("SafeKid AI — Processing Throughput (FPS) Before vs After Optimization", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['scenario']}\n({r['resolution']})" for r in comparison_records], fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=11)
    ax.set_ylim(0, 9.0)

    for bar, val in zip(rects1, base_fps):
        ax.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val, pct in zip(rects2, opt_fps, [r["improvements"]["fps_gain_pct"] for r in comparison_records]):
        ax.annotate(f"{val:.2f}\n(+{pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0369a1")

    fig.tight_layout()
    fps_plot_path = out_dir / "prompt_17_fps_comparison.png"
    plt.savefig(fps_plot_path)
    plt.close()
    print(f"[Saved FPS Plot] {fps_plot_path}")

    # Plot 2: Latency Breakdown (Stacked Stages)
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    x_indices = np.arange(len(scenarios) * 2)
    labels = []
    yolo_times = []
    vis_times = []
    track_times = []
    other_times = []

    for b, o in zip(baseline_data, optimized_data):
        sc = b["name"]
        labels.extend([f"{sc}\n[Base]", f"{sc}\n[Opt]"])
        for d in (b, o):
            sm = d["stage_metrics"]
            yolo = sm["yolo_inference_ms"]["mean"]
            vis = sm["vis_and_writing_ms"]["mean"]
            track = sm["bytetrack_ms"]["mean"]
            tot = sm["total_frame_ms"]["mean"]
            other = max(0.0, tot - (yolo + vis + track))
            yolo_times.append(yolo)
            vis_times.append(vis)
            track_times.append(track)
            other_times.append(other)

    bar_width = 0.6
    p1 = ax.bar(x_indices, yolo_times, bar_width, label="YOLOv8n Inference", color="#f97316")
    p2 = ax.bar(x_indices, vis_times, bar_width, bottom=yolo_times, label="Vis & Video Writing", color="#3b82f6")
    p3 = ax.bar(x_indices, track_times, bar_width, bottom=np.array(yolo_times) + np.array(vis_times), label="ByteTrack Tracking", color="#10b981")
    p4 = ax.bar(x_indices, other_times, bar_width, bottom=np.array(yolo_times) + np.array(vis_times) + np.array(track_times), label="Dec / ROI / Safety / Incident", color="#8b5cf6")

    ax.set_ylabel("Mean Frame Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("SafeKid AI — Latency Breakdown by Stage (ms) [Lower is Better]", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=11, loc="upper right")
    ax.set_ylim(0, 240)

    fig.tight_layout()
    latency_plot_path = out_dir / "prompt_17_latency_breakdown.png"
    plt.savefig(latency_plot_path)
    plt.close()
    print(f"[Saved Latency Breakdown Plot] {latency_plot_path}")

if __name__ == "__main__":
    generate_comparison()

