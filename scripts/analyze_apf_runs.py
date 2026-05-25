#!/usr/bin/env python3
"""Analyze APF ablation runs and generate comparison report."""

from __future__ import annotations

import argparse
import json
import glob
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

# ------------------------------------------------------------------------------
# TensorBoard availability
# ------------------------------------------------------------------------------
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    HAS_TENSORBOARD = True
except Exception:
    HAS_TENSORBOARD = False

# ------------------------------------------------------------------------------
# Config parsing helpers
# ------------------------------------------------------------------------------


def extract_first_json(text: str) -> dict[str, Any] | None:
    """Extract the first top-level JSON object from text via brace counting."""
    start = text.find("{")
    if start == -1:
        return None
    count = 0
    end = start
    for i, c in enumerate(text[start:], start):
        if c == "{":
            count += 1
        elif c == "}":
            count -= 1
        if count == 0:
            end = i
            break
    if count != 0:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def scan_runs(logs_dir: Path) -> list[dict[str, Any]]:
    """Scan logs directory and extract run metadata from config.txt."""
    runs: list[dict[str, Any]] = []
    pattern = str(logs_dir / "*" / "config" / "config.txt")
    for config_path in sorted(glob.glob(pattern)):
        try:
            text = Path(config_path).read_text(encoding="utf-8")
            data = extract_first_json(text)
            if data is None:
                continue
            run_dir = Path(config_path).parent.parent
            has_tfevents = bool(list(run_dir.glob("events.out.tfevents.*")))
            runs.append(
                {
                    "name": run_dir.name,
                    "dir": run_dir,
                    "enable_apf": data.get("enable_apf", False),
                    "att": data.get("apf_attractive_weight", 0.5),
                    "rep": data.get("apf_repulsive_weight", -0.5),
                    "timesteps": data.get("timesteps", 0),
                    "num_envs": data.get("num_envs", 0),
                    "seed": data.get("seed", 42),
                    "has_tfevents": has_tfevents,
                }
            )
        except Exception:
            continue
    return runs


def group_runs(runs: list[dict[str, Any]]) -> dict[tuple[bool, float, float], dict[str, Any]]:
    """Group runs by (enable_apf, att, rep) and keep the latest per group."""
    groups: dict[tuple[bool, float, float], list[dict[str, Any]]] = {}
    for r in runs:
        key = (bool(r["enable_apf"]), float(r["att"]), float(r["rep"]))
        groups.setdefault(key, []).append(r)
    latest: dict[tuple[bool, float, float], dict[str, Any]] = {}
    for key, grp in groups.items():
        grp.sort(key=lambda x: x["name"], reverse=True)
        latest[key] = grp[0]
    return latest


# ------------------------------------------------------------------------------
# TensorBoard data extraction
# ------------------------------------------------------------------------------


def read_scalar(tfevents_path: Path, tag: str) -> tuple[list[int], list[float]] | None:
    """Read a scalar tag from a tfevents file."""
    if not HAS_TENSORBOARD:
        return None
    try:
        ea = EventAccumulator(str(tfevents_path))
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return None
        events = ea.Scalars(tag)
        return [e.step for e in events], [e.value for e in events]
    except Exception:
        return None


def plot_comparison(
    latest_runs: dict[tuple[bool, float, float], dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Generate comparison plots from TensorBoard data."""
    if not HAS_TENSORBOARD:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags_of_interest = [
        "Train/Reward/episode_total/mean",
        "Train/Reward/episode_total/max",
        "Train/Reward/instantaneous/mean",
        "Termination/reached_goal/ratio_window",
        "Termination/collision/ratio_window",
        "Termination/time_out/ratio_window",
        "Train/Episode/total_timesteps/mean",
        "Train/Learning/lr",
    ]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # Plot each tag individually for clarity
    for tag in tags_of_interest:
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted_any = False
        for key in sorted(latest_runs.keys()):
            r = latest_runs[key]
            tfevents = list(r["dir"].glob("events.out.tfevents.*"))
            if not tfevents:
                continue
            data = read_scalar(tfevents[0], tag)
            if data is None:
                continue
            steps, vals = data
            if len(steps) == 0:
                continue
            label = "baseline" if not r["enable_apf"] else f"att={r['att']} rep={r['rep']}"
            ax.plot(steps, vals, label=label, alpha=0.8, linewidth=1.2)
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_xlabel("Step")
        ax.set_ylabel(tag.split("/")[-1])
        ax.set_title(tag)
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = tag.replace("/", "_")
        fig_path = figures_dir / f"{safe_name}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        generated.append(fig_path)

    return generated


def extract_final_metrics(
    latest_runs: dict[tuple[bool, float, float], dict[str, Any]],
) -> dict[tuple[bool, float, float], dict[str, float | None]]:
    """Extract final values for key metrics from each run."""
    tags = [
        "Train/Reward/episode_total/mean",
        "Termination/reached_goal/ratio_window",
        "Termination/collision/ratio_window",
        "Termination/time_out/ratio_window",
    ]
    metrics: dict[tuple[bool, float, float], dict[str, float | None]] = {}
    for key, r in latest_runs.items():
        metrics[key] = {}
        tfevents = list(r["dir"].glob("events.out.tfevents.*"))
        if not tfevents:
            for tag in tags:
                metrics[key][tag] = None
            continue
        for tag in tags:
            data = read_scalar(tfevents[0], tag)
            if data is None or len(data[1]) == 0:
                metrics[key][tag] = None
            else:
                metrics[key][tag] = data[1][-1]
    return metrics


# ------------------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------------------


def generate_report(
    latest_runs: dict[tuple[bool, float, float], dict[str, Any]],
    output_dir: Path,
    logs_dir: Path,
) -> Path:
    """Generate Markdown comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"

    lines: list[str] = []
    lines.append("# APF Ablation Study Report\n")
    lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
    lines.append(f"**Total unique experiments:** {len(latest_runs)}\n")

    # Parameters table
    lines.append("## Experiment Parameters\n")
    lines.append(
        "| Group | APF | Att Weight | Rep Weight | Timesteps | Envs | Seed | Run Name | Has Events |"
    )
    lines.append(
        "|-------|-----|------------|------------|-----------|------|------|----------|------------|"
    )

    for key in sorted(latest_runs.keys()):
        r = latest_runs[key]
        group = "baseline" if not r["enable_apf"] else f"apf_{r['att']}_{r['rep']}"
        apf_str = "off" if not r["enable_apf"] else "on"
        lines.append(
            f"| {group} | {apf_str} | {r['att']} | {r['rep']} | {r['timesteps']} | "
            f"{r['num_envs']} | {r['seed']} | `{r['name']}` | {r['has_tfevents']} |"
        )

    # Final metrics table (if tensorboard available)
    if HAS_TENSORBOARD:
        metrics = extract_final_metrics(latest_runs)
        lines.append("\n## Final Metrics (Last Recorded Value)\n")
        lines.append(
            "| Group | Reward Mean | Goal Ratio | Collision Ratio | Timeout Ratio |"
        )
        lines.append(
            "|-------|-------------|------------|-----------------|---------------|"
        )
        for key in sorted(latest_runs.keys()):
            r = latest_runs[key]
            group = "baseline" if not r["enable_apf"] else f"apf_{r['att']}_{r['rep']}"
            m = metrics.get(key, {})
            reward = f"{m.get('Train/Reward/episode_total/mean'):.4f}" if m.get("Train/Reward/episode_total/mean") is not None else "N/A"
            goal = f"{m.get('Termination/reached_goal/ratio_window'):.4f}" if m.get("Termination/reached_goal/ratio_window") is not None else "N/A"
            collision = f"{m.get('Termination/collision/ratio_window'):.4f}" if m.get("Termination/collision/ratio_window") is not None else "N/A"
            timeout = f"{m.get('Termination/time_out/ratio_window'):.4f}" if m.get("Termination/time_out/ratio_window") is not None else "N/A"
            lines.append(
                f"| {group} | {reward} | {goal} | {collision} | {timeout} |"
            )

    # Log directories
    lines.append("\n## Log Directories\n")
    for key in sorted(latest_runs.keys()):
        r = latest_runs[key]
        group = "baseline" if not r["enable_apf"] else f"apf_{r['att']}_{r['rep']}"
        lines.append(f"- **{group}**: `{r['dir']}`")

    # TensorBoard instructions
    lines.append("\n## TensorBoard Viewing\n")
    lines.append("To launch TensorBoard for interactive exploration:\n")
    lines.append("```bash")
    lines.append(f"tensorboard --logdir=\"{logs_dir}\"")
    lines.append("```")

    if not HAS_TENSORBOARD:
        lines.append(
            "\n> **Note:** `tensorboard` Python package is not installed in the current environment. "
            "Install it (`pip install tensorboard`) and re-run this script to generate training curves "
            "and extract final metrics automatically.\n"
        )
    else:
        lines.append(
            "\nTraining curve figures have been saved to the `figures/` subdirectory.\n"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze APF ablation runs and generate comparison report."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="../logs",
        help="Path to the logs directory (default: ../logs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../logs/apf_analysis",
        help="Path to the output report directory (default: ../logs/apf_analysis)",
    )
    args = parser.parse_args()

    logs_dir = Path(args.log_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not logs_dir.exists():
        print(f"[ERROR] Log directory not found: {logs_dir}")
        sys.exit(1)

    print(f"[INFO] Scanning logs in: {logs_dir}")
    runs = scan_runs(logs_dir)
    print(f"[INFO] Found {len(runs)} runs")

    if len(runs) == 0:
        print("[WARN] No runs found. Have any experiments completed yet?")
        sys.exit(0)

    latest = group_runs(runs)
    print(f"[INFO] Unique experiment configurations: {len(latest)}")

    # Generate plots if tensorboard is available
    if HAS_TENSORBOARD:
        print("[INFO] TensorBoard detected. Generating comparison figures...")
        figures = plot_comparison(latest, output_dir)
        print(f"[INFO] Generated {len(figures)} figures in: {output_dir / 'figures'}")
    else:
        print("[INFO] TensorBoard not available. Skipping figure generation.")

    # Generate report
    report_path = generate_report(latest, output_dir, logs_dir)
    print(f"[INFO] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
