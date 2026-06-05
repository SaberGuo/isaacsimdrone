#!/usr/bin/env python3
"""APF vs Baseline training comparison analysis."""

import json, os, glob, re, sys
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

TAGS = [
    'Train/Reward/episode_total/mean',
    'Train/Reward/episode_total/max',
    'Termination/reached_goal/ratio_window',
    'Termination/collision/ratio_window',
    'Termination/time_out/ratio_window',
    'Train/Episode/total_timesteps/mean',
]

# Shorter display names
TAG_LABELS = {
    'Train/Reward/episode_total/mean':         'Episode Reward (mean)',
    'Train/Reward/episode_total/max':          'Episode Reward (max)',
    'Termination/reached_goal/ratio_window':   'Success Rate',
    'Termination/collision/ratio_window':      'Collision Rate',
    'Termination/time_out/ratio_window':       'Timeout Rate',
    'Train/Episode/total_timesteps/mean':      'Episode Length (steps)',
}


def load_config(run_dir):
    cfg_path = os.path.join(run_dir, 'config', 'config.txt')
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path) as f:
        content = f.read()
    m = re.search(r'\{.*?\}', content, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


def load_scalars(run_dir, tags):
    event_files = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
    if not event_files:
        return {}
    ea = event_accumulator.EventAccumulator(
        event_files[0],
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    available = set(ea.Tags().get('scalars', []))
    data = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps  = np.array([e.step  for e in events])
            values = np.array([e.value for e in events])
            data[tag] = (steps, values)
    return data


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='same')


def final_stats(steps, values, last_frac=0.1):
    """Mean/std over the final fraction of training."""
    n = max(1, int(len(values) * last_frac))
    tail = values[-n:]
    return float(np.mean(tail)), float(np.std(tail))


def scan_runs():
    runs = sorted(glob.glob(os.path.join(LOG_DIR, '26-*_PPO')))
    result = []
    for d in runs:
        cfg = load_config(d)
        if not cfg:
            continue
        result.append({
            'dir':      d,
            'name':     os.path.basename(d),
            'enable_apf':           cfg.get('enable_apf', False),
            'apf_attractive_weight': cfg.get('apf_attractive_weight', 0.0),
            'apf_repulsive_weight':  cfg.get('apf_repulsive_weight',  0.0),
            'timesteps':            cfg.get('timesteps', 0),
            'num_envs':             cfg.get('num_envs', 0),
        })
    return result


def group_by_condition(runs):
    groups = defaultdict(list)
    for r in runs:
        key = (r['enable_apf'],
               r['apf_attractive_weight'],
               r['apf_repulsive_weight'],
               r['timesteps'],
               r['num_envs'])
        groups[key].append(r)
    # Keep latest run per condition
    best = {}
    for key, rs in groups.items():
        best[key] = sorted(rs, key=lambda x: x['name'])[-1]
    return best


def condition_label(r):
    if not r['enable_apf']:
        return 'Baseline (no APF)'
    att = r['apf_attractive_weight']
    rep = r['apf_repulsive_weight']
    return f'APF att={att} rep={rep}'


def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    all_runs = scan_runs()
    if not all_runs:
        print('No runs with config found.')
        sys.exit(1)

    conditions = group_by_condition(all_runs)

    print(f'Found {len(all_runs)} total runs → {len(conditions)} distinct conditions')
    print()

    # Load data per condition
    cond_data = {}
    for key, r in sorted(conditions.items()):
        label = condition_label(r)
        scalars = load_scalars(r['dir'], TAGS)
        cond_data[label] = {'run': r, 'scalars': scalars}
        n_tags = len(scalars)
        max_step = max((s[-1] for s, _ in scalars.values()), default=0)
        print(f'  [{label}]  run={r["name"]}  tags={n_tags}  max_step={max_step:,}')

    print()

    # --- Text summary table ---
    print('=' * 90)
    print('FINAL PERFORMANCE COMPARISON  (average over last 10% of training)')
    print('=' * 90)

    header_tags = [
        'Train/Reward/episode_total/mean',
        'Termination/reached_goal/ratio_window',
        'Termination/collision/ratio_window',
        'Termination/time_out/ratio_window',
        'Train/Episode/total_timesteps/mean',
    ]
    col_w = 16
    header = f'{"Condition":<35}' + ''.join(f'{TAG_LABELS[t][:col_w-1]:>{col_w}}' for t in header_tags)
    print(header)
    print('-' * len(header))

    rows = []
    for label, d in sorted(cond_data.items()):
        scalars = d['scalars']
        cols = [f'{label:<35}']
        stats_row = {'label': label}
        for t in header_tags:
            if t in scalars:
                mean_val, std_val = final_stats(*scalars[t])
                stats_row[t] = mean_val
                cols.append(f'{mean_val:>{col_w}.3f}')
            else:
                stats_row[t] = None
                cols.append(f'{"N/A":>{col_w}}')
        print(''.join(cols))
        rows.append(stats_row)

    print()

    # --- Delta analysis (APF vs best baseline) ---
    baseline_rows = [r for r in rows if 'Baseline' in r['label']]
    apf_rows      = [r for r in rows if 'Baseline' not in r['label']]

    if baseline_rows and apf_rows:
        bl = baseline_rows[-1]  # use most recent baseline
        print('DELTA vs BASELINE (latest baseline run)')
        print('-' * 60)
        for r in apf_rows:
            print(f'  {r["label"]}')
            for t in header_tags:
                if bl.get(t) is not None and r.get(t) is not None:
                    delta = r[t] - bl[t]
                    pct   = (delta / (abs(bl[t]) + 1e-9)) * 100
                    sign  = '+' if delta >= 0 else ''
                    tag_s = TAG_LABELS[t]
                    print(f'    {tag_s:<35}  {sign}{delta:+.3f}  ({sign}{pct:+.1f}%)')
            print()

    # --- Plots ---
    if HAS_MPL and len(cond_data) >= 2:
        plot_tags = [
            'Train/Reward/episode_total/mean',
            'Termination/reached_goal/ratio_window',
            'Termination/collision/ratio_window',
        ]
        fig, axes = plt.subplots(1, len(plot_tags), figsize=(6 * len(plot_tags), 4))
        if len(plot_tags) == 1:
            axes = [axes]

        colors = plt.cm.tab10.colors
        for ax, tag in zip(axes, plot_tags):
            for i, (label, d) in enumerate(sorted(cond_data.items())):
                if tag not in d['scalars']:
                    continue
                steps, values = d['scalars'][tag]
                # Normalise steps to millions
                x = steps / 1e6
                y = smooth(values, window=30)
                lw = 2.0 if 'Baseline' not in label else 1.5
                ls = '-' if 'Baseline' not in label else '--'
                ax.plot(x, y, label=label, color=colors[i % len(colors)], lw=lw, ls=ls)
            ax.set_title(TAG_LABELS.get(tag, tag))
            ax.set_xlabel('Timesteps (M)')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir = os.path.join(LOG_DIR, '..', 'figures')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'apf_vs_baseline.png')
        plt.savefig(out_path, dpi=150)
        print(f'Plot saved → {out_path}')
    elif not HAS_MPL:
        print('(matplotlib not available, skipping plots)')


if __name__ == '__main__':
    main()
