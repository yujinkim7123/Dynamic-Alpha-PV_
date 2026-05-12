#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore', message='findfont')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

plt.rcParams.update({
    'figure.facecolor':  '#0a0e1a',
    'axes.facecolor':    '#111827',
    'axes.edgecolor':    '#1e2d45',
    'axes.labelcolor':   '#94a3b8',
    'axes.titlecolor':   '#e2e8f0',
    'text.color':        '#e2e8f0',
    'xtick.color':       '#64748b',
    'ytick.color':       '#64748b',
    'grid.color':        '#1e2d45',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'legend.facecolor':  '#111827',
    'legend.edgecolor':  '#1e2d45',
    'legend.labelcolor': '#94a3b8',
    'font.size':         10,
})

DIM_COLOR = {
    'conscientiousness': '#00d4ff',
    'extraversion':      '#51cf66',
    'neuroticism':       '#ff6b6b',
    'agreeableness':     '#ffd43b',
    'openness':          '#c084fc',
}

DIM_SHORT = {
    'conscientiousness': 'CON',
    'extraversion':      'EXT',
    'neuroticism':       'NEU',
    'agreeableness':     'AGR',
    'openness':          'OPN',
}

BUHLER_DELTA = {
    'first_job':        {'conscientiousness': +0.276, 'neuroticism': -0.232, 'extraversion': +0.222},
    'new_relationship': {'conscientiousness': +0.154, 'neuroticism': -0.337, 'extraversion': +0.337},
    'marriage':         {'openness': -0.175, 'neuroticism': -0.066, 'extraversion': +0.066},
    'divorce':          {'conscientiousness': +0.096, 'neuroticism': +0.052, 'extraversion': -0.052},
    'graduation':       {'neuroticism': -0.164, 'extraversion': +0.133},
    'unemployment':     {'neuroticism': -0.095, 'conscientiousness': -0.058},
}

DIMS = ['conscientiousness', 'extraversion', 'neuroticism', 'agreeableness', 'openness']


# =============================================================================
# 1. 데이터 로드
# =============================================================================
def load_summary(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'life_event' not in data and 'experiment' in data:
        data['life_event'] = data['experiment']

    if 'buhler_delta' not in data and 'combined_buhler' in data:
        SHORT_TO_FULL = {
            'CON': 'conscientiousness', 'EXT': 'extraversion',
            'NEU': 'neuroticism',       'AGR': 'agreeableness',
            'OPN': 'openness',
        }
        _SKIP_DIMS = {'emotional_stability', 'emotional stability'}
        data['buhler_delta'] = {
            SHORT_TO_FULL.get(dim, dim): info['d_sum']
            for dim, info in data['combined_buhler'].items()
            if dim not in _SKIP_DIMS
        }

    if not data.get('buhler_delta'):
        event = data.get('life_event', '')
        data['buhler_delta'] = BUHLER_DELTA.get(event, {})

    event  = data.get('life_event', '—')
    events = data.get('events_detected', [event] if event != '—' else [])
    n_ckpt = len(data.get('bfi_history', []))

    print(f"✅ 로드 완료: {path}")
    print(f"   감지 사건  : {events if events else '—'}")
    print(f"   체크포인트 : {n_ckpt}개")
    print(f"   buhler_delta: {data.get('buhler_delta', {})}")
    return data


def load_from_dir(result_dir: str, event: str) -> dict:
    event_dir    = Path(result_dir) / event
    summary_path = event_dir / 'summary.json'

    if summary_path.exists():
        return load_summary(str(summary_path))

    ckpt_dirs = sorted(event_dir.glob('checkpoint_*'))
    bfi_history, liwc_history = [], []
    for cd in ckpt_dirs:
        bfi_p  = cd / 'bfi_scores.json'
        liwc_p = cd / 'liwc_scores.json'
        if bfi_p.exists():
            with open(bfi_p) as f: bfi_history.append(json.load(f))
        if liwc_p.exists():
            with open(liwc_p) as f: liwc_history.append(json.load(f))

    buhler = BUHLER_DELTA.get(event, {})
    data = {
        'life_event':   event,
        'buhler_delta': buhler,
        'bfi_history':  bfi_history,
        'liwc_history': liwc_history,
        'checkpoints':  [str(cd.name) for cd in ckpt_dirs],
    }
    print(f"✅ 체크포인트 {len(bfi_history)}개 조합 완료")
    return data


# =============================================================================
# 2. 데이터프레임 변환
# =============================================================================
def to_dataframe(data: dict) -> tuple:
    bfi_df  = pd.DataFrame(data.get('bfi_history', []))
    liwc_df = pd.DataFrame(data.get('liwc_history', []))

    n = len(bfi_df)
    bfi_df.index  = [f'CP{str(i).zfill(2)}' for i in range(n)]
    if len(liwc_df):
        liwc_df.index = [f'CP{str(i).zfill(2)}' for i in range(len(liwc_df))]

    return bfi_df, liwc_df


# =============================================================================
# 2-1. 지수평활 (EMA)
# =============================================================================
def exponential_smooth(series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    수식: s_t = alpha × x_t + (1 - alpha) × s_{t-1}
    alpha=0.3 : 완만한 평활 (논문용 권장)
    alpha=1.0 : 평활 없음 (원본 그대로)
    """
    s = np.empty_like(series, dtype=float)
    s[0] = series[0]
    for t in range(1, len(series)):
        s[t] = alpha * series[t] + (1 - alpha) * s[t - 1]
    return s


def smooth_dataframe(df: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    smoothed = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        smoothed[col] = exponential_smooth(df[col].values.astype(float), alpha)
    return smoothed


# =============================================================================
# 3. 평가지표 계산
# =============================================================================
def pearson_r(x, y):
    x, y = np.array(x), np.array(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(data: dict) -> dict:
    bfi_df, liwc_df = to_dataframe(data)

    _SKIP_DIMS = {'emotional_stability', 'emotional stability'}

    buhler = data.get('buhler_delta')
    if not buhler and 'combined_buhler' in data:
        SHORT_TO_FULL = {
            'CON': 'conscientiousness', 'EXT': 'extraversion',
            'NEU': 'neuroticism',       'AGR': 'agreeableness',
            'OPN': 'openness',
        }
        buhler = {
            SHORT_TO_FULL.get(dim, dim): info['d_sum']
            for dim, info in data['combined_buhler'].items()
            if dim not in _SKIP_DIMS
        }
    if not buhler:
        event  = data.get('life_event') or data.get('experiment', '')
        buhler = BUHLER_DELTA.get(event, {})

    for _skip in _SKIP_DIMS:
        buhler.pop(_skip, None)

    if len(bfi_df) < 2:
        print("⚠️  체크포인트가 2개 이상 필요합니다.")
        return {}

    base  = bfi_df.iloc[0]
    final = bfi_df.iloc[-1]

    dm = {}
    for dim, d in buhler.items():
        change = final.get(dim, np.nan) - base.get(dim, np.nan)
        if d == 0 or np.isnan(change): dm[dim] = None
        else: dm[dim] = (change > 0) == (d > 0)

    active_dm = {k: v for k, v in dm.items() if v is not None}
    dm_rate   = sum(bool(v) for v in active_dm.values()) / len(active_dm) if active_dm else 0.0

    pr = {}
    for dim in DIMS:
        bfi_col  = dim
        liwc_col = f'{dim}_liwc'
        if bfi_col not in bfi_df.columns: continue
        bfi_vals  = bfi_df[bfi_col].tolist()
        liwc_vals = liwc_df[liwc_col].tolist() if (len(liwc_df) and liwc_col in liwc_df.columns) else [np.nan]*len(bfi_vals)
        bfi_delta  = [bfi_vals[i+1]  - bfi_vals[i]  for i in range(len(bfi_vals)-1)]
        liwc_delta = [liwc_vals[i+1] - liwc_vals[i] for i in range(len(liwc_vals)-1)]
        pr[dim] = pearson_r(bfi_delta, liwc_delta)

    deltas = {dim: float(final.get(dim, np.nan) - base.get(dim, np.nan)) for dim in DIMS}

    BFI_STD = 1.0
    effect_size = {}
    for dim in DIMS:
        if dim not in bfi_df.columns:
            effect_size[dim] = float('nan')
            continue
        delta_val = float(final.get(dim, np.nan) - base.get(dim, np.nan))
        effect_size[dim] = float('nan') if np.isnan(delta_val) else round(delta_val / BFI_STD, 4)

    cancel_accuracy = None
    cancel_detail   = {}
    if data.get('combined_buhler'):
        SHORT_TO_FULL = {
            'CON': 'conscientiousness', 'EXT': 'extraversion',
            'NEU': 'neuroticism',       'AGR': 'agreeableness',
            'OPN': 'openness',
        }
        cancel_dims    = []
        cancel_correct = 0
        for dim_key, info in data['combined_buhler'].items():
            if info.get('relation') != 'cancel': continue
            full_dim = SHORT_TO_FULL.get(dim_key, dim_key)
            d_sum    = info.get('d_sum', 0)
            actual   = deltas.get(full_dim, float('nan'))
            if np.isnan(actual) or d_sum == 0:
                cancel_detail[full_dim] = None
                continue
            correct = (actual > 0) == (d_sum > 0)
            cancel_detail[full_dim] = {'d_sum': d_sum, 'actual': actual, 'correct': correct}
            cancel_dims.append(full_dim)
            if correct: cancel_correct += 1
        cancel_accuracy = (cancel_correct / len(cancel_dims)) if cancel_dims else None

    return {
        'direction_match':      dm,
        'direction_match_rate': dm_rate,
        'pearson_r':            pr,
        'effect_size':          effect_size,
        'cancel_accuracy':      cancel_accuracy,
        'cancel_detail':        cancel_detail,
        'bfi_base':             base.to_dict(),
        'bfi_final':            final.to_dict(),
        'bfi_delta':            deltas,
        'buhler_delta':         buhler,
    }


def print_metrics(metrics: dict, data: dict = None):
    print("\n" + "="*60)
    print("📊 평가지표 요약")
    print("="*60)

    if data:
        events = data.get('events_detected', [])
        if events:
            print(f"\n 감지된 사건: {events}")

    print(f"\n Direction Match Rate: {metrics['direction_match_rate']:.1%}")

    if data and data.get('combined_direction_match_rate') is not None:
        rate = data['combined_direction_match_rate']
        print(f" Combined Match Rate : {rate:.1%}")
        if data.get('combined_buhler'):
            print(f"\n 복합 사건 방향성:")
            print(f"   {'차원':>4}  {'관계':>10}  {'d_sum':>8}  {'사건'}")
            SHORT_TO_FULL = {
                'CON': 'conscientiousness', 'EXT': 'extraversion',
                'NEU': 'neuroticism',       'AGR': 'agreeableness',
                'OPN': 'openness',
            }
            for dim, info in data['combined_buhler'].items():
                if dim in {'emotional_stability', 'emotional stability'}: continue
                full  = SHORT_TO_FULL.get(dim, dim)
                short = DIM_SHORT.get(full, dim)
                print(f"   {short:>4}  {info['relation']:>10}  {info['d_sum']:>+8.3f}  {info['events']}")

    for dim, match in metrics['direction_match'].items():
        icon  = '✅' if match == True else ('❌' if match == False else '—')
        d     = metrics['buhler_delta'].get(dim, 0)
        delta = metrics['bfi_delta'].get(dim, 0)
        print(f"   {DIM_SHORT[dim]:>3}: {icon}  Expected={d:+.3f}  Actual={delta:+.3f}")

    print(f"\n📈 Pearson r (BFI변화 ↔ LIWC변화):")
    for dim, r in metrics['pearson_r'].items():
        bar = '█' * int(abs(r) * 10) if not np.isnan(r) else ''
        print(f"   {DIM_SHORT[dim]:>3}: {r:+.4f}  {bar}")

    print(f"\n📐 Effect Size (Cohen's d):")
    for dim in DIMS:
        d_val = metrics['effect_size'].get(dim, float('nan'))
        if np.isnan(d_val): continue
        if   abs(d_val) >= 0.8: label = 'large'
        elif abs(d_val) >= 0.5: label = 'medium'
        elif abs(d_val) >= 0.2: label = 'small'
        else:                   label = 'negligible'
        print(f"   {DIM_SHORT[dim]:>3}: {d_val:+.4f}  ({label})")

    if metrics.get('cancel_accuracy') is not None:
        print(f"\n🎯 Cancel Accuracy (충돌 차원):")
        print(f"   전체: {metrics['cancel_accuracy']:.1%}")
        for dim, detail in metrics.get('cancel_detail', {}).items():
            if detail is None: continue
            icon = '✅' if detail['correct'] else '❌'
            print(f"   {DIM_SHORT.get(dim, dim):>3}: {icon}  d_sum={detail['d_sum']:+.3f}  actual={detail['actual']:+.3f}")

    print("="*60)


# =============================================================================
# 4. 시각화
# =============================================================================

def _set_bfi_yaxis(ax, vals):
    """BFI y축 동적 범위 공통 설정"""
    if vals:
        y_min = max(1.0, min(vals) - 0.3)
        y_max = min(5.0, max(vals) + 0.3)
        ax.set_ylim(y_min, y_max)
        minor = np.arange(round(y_min, 2), round(y_max, 2) + 0.05, 0.05)
        ax.set_yticks(minor, minor=True)
        major = np.arange(round(y_min * 2) / 2, round(y_max * 2) / 2 + 0.5, 0.5)
        ax.set_yticks(major, minor=False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.grid(True, which='minor', alpha=0.1)
        ax.grid(True, which='major', alpha=0.3)
    else:
        ax.set_ylim(1, 5)


# ── 4-1. BFI 시계열 (원본) ──
def plot_bfi_raw(bfi_df: pd.DataFrame, buhler: dict, title: str = ''):
    """원본 BFI 값 그대로 시각화 (평활 없음)"""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0a0e1a')

    all_vals = []
    for dim in DIMS:
        if dim not in bfi_df.columns: continue
        vals  = bfi_df[dim].values
        color = DIM_COLOR[dim]
        ax.plot(bfi_df.index, vals, color=color, linewidth=2,
                marker='o', markersize=5, label=DIM_SHORT[dim])
        if dim in buhler:
            d = buhler[dim]
            ax.annotate(f"{'↑' if d>0 else '↓'}({d:+.3f})",
                        xy=(bfi_df.index[-1], vals[-1]),
                        xytext=(8, 0), textcoords='offset points',
                        color=color, fontsize=9, va='center')
        all_vals.extend(bfi_df[dim].dropna().tolist())

    _set_bfi_yaxis(ax, all_vals)
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('BFI Score')
    ax.set_title(title or 'BFI Score Trajectory — Raw', pad=12)
    ax.legend(loc='upper left', framealpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


# ── 4-2. BFI 시계열 (EMA 평활) ──
def plot_bfi_smoothed(bfi_df: pd.DataFrame, buhler: dict, title: str = '',
                      smooth_alpha: float = 0.3):
    """EMA 평활 BFI 시계열 (원본은 반투명 점선으로 함께 표시)"""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0a0e1a')

    bfi_smoothed = smooth_dataframe(bfi_df, alpha=smooth_alpha)
    all_vals = []

    for dim in DIMS:
        if dim not in bfi_df.columns: continue
        raw_vals    = bfi_df[dim].values
        smooth_vals = bfi_smoothed[dim].values
        color = DIM_COLOR[dim]

        # 원본: 반투명 점선
        ax.plot(bfi_df.index, raw_vals, color=color, linewidth=1,
                linestyle='--', alpha=0.3, marker='o', markersize=3)
        # EMA: 실선 (메인)
        ax.plot(bfi_df.index, smooth_vals, color=color, linewidth=2,
                marker='o', markersize=5, label=DIM_SHORT[dim])

        if dim in buhler:
            d = buhler[dim]
            ax.annotate(f"{'↑' if d>0 else '↓'}({d:+.3f})",
                        xy=(bfi_df.index[-1], smooth_vals[-1]),
                        xytext=(8, 0), textcoords='offset points',
                        color=color, fontsize=9, va='center')

        all_vals.extend(bfi_df[dim].dropna().tolist())

    _set_bfi_yaxis(ax, all_vals)
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('BFI Score')
    ax.set_title(title or f'BFI Score Trajectory — EMA (α={smooth_alpha})', pad=12)
    ax.legend(loc='upper left', framealpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


# ── 4-3. 방향 일치 바차트 ──
def plot_direction_match(metrics: dict):
    buhler = metrics['buhler_delta']
    delta  = metrics['bfi_delta']
    dm     = metrics['direction_match']

    active_dims = [d for d in DIMS if d in buhler]
    if not active_dims:
        print("⚠️  활성 차원 없음")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0e1a')

    ax = axes[0]
    x  = np.arange(len(active_dims))
    w  = 0.35
    buhler_vals = [buhler[d] for d in active_dims]
    delta_vals  = [delta.get(d, 0) for d in active_dims]
    colors_dim  = [DIM_COLOR[d] for d in active_dims]

    ax.bar(x - w/2, buhler_vals, w, label='Buhler d (Expected)',
           color=[c + '60' for c in colors_dim], edgecolor=colors_dim, linewidth=1.5)
    ax.bar(x + w/2, delta_vals,  w, label='실제 BFI 변화',
           color=colors_dim, alpha=0.85)

    ax.axhline(0, color='#64748b', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_SHORT[d] for d in active_dims])
    ax.set_ylabel('Delta')
    ax.set_title('Buhler Expected vs Actual BFI Delta', pad=10)
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    matched   = sum(1 for v in dm.values() if v == True)
    unmatched = sum(1 for v in dm.values() if v == False)
    na        = sum(1 for v in dm.values() if v is None)
    sizes  = [matched, unmatched]
    labels = [f'Match ({matched})', f'Mismatch ({unmatched})']
    colors = ['#51cf66', '#ff6b6b']
    if na > 0:
        sizes.append(na); labels.append(f'N/A ({na})'); colors.append('#64748b')

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90,
        wedgeprops={'edgecolor': '#0a0e1a', 'linewidth': 2}
    )
    for t in autotexts: t.set_color('#0a0e1a'); t.set_fontweight('bold')
    ax2.set_title(f'Direction Match Rate: {metrics["direction_match_rate"]:.1%}', pad=10)
    plt.tight_layout()
    return fig


# ── 4-4. Pearson r 바차트 ──
def plot_pearson_r(metrics: dict):
    pr   = metrics['pearson_r']
    dims = [d for d in DIMS if d in pr and not np.isnan(pr[d])]
    if not dims:
        print("⚠️  Pearson r 데이터 없음")
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0a0e1a')

    vals   = [pr[d] for d in dims]
    colors = [DIM_COLOR[d] for d in dims]
    bars   = ax.barh([DIM_SHORT[d] for d in dims], vals,
                     color=colors, alpha=0.85, edgecolor=colors, linewidth=1)

    ax.axvline(0,    color='#64748b', linewidth=0.8, linestyle='--')
    ax.axvline(0.6,  color='#51cf66', linewidth=1.0, linestyle=':', alpha=0.5, label='r=0.6 기준선')
    ax.axvline(-0.6, color='#ff6b6b', linewidth=1.0, linestyle=':', alpha=0.5)

    for bar, val in zip(bars, vals):
        x = val + (0.02 if val >= 0 else -0.02)
        ax.text(x, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center',
                ha='left' if val >= 0 else 'right',
                color='#e2e8f0', fontsize=10, fontweight='bold')

    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel('Pearson r')
    ax.set_title('Pearson r — BFI Delta vs LIWC Delta', pad=10)
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig


# ── 4-5. LIWC 시계열 (원본) ──
def plot_liwc_raw(liwc_df: pd.DataFrame, title: str = ''):
    """원본 LIWC 값 그대로 시각화"""
    if liwc_df.empty:
        print("⚠️  LIWC 데이터 없음")
        return None

    liwc_cols = {
        'conscientiousness_liwc': ('CON-Achieve', '#00d4ff'),
        'extraversion_liwc':      ('EXT-Social',  '#51cf66'),
        'neuroticism_liwc':       ('NEU-NegEmo',  '#ff6b6b'),
        'agreeableness_liwc':     ('AGR-PosEmo',  '#ffd43b'),
        'openness_liwc':          ('OPN-Insight', '#c084fc'),
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0a0e1a')

    for col, (label, color) in liwc_cols.items():
        if col not in liwc_df.columns: continue
        ax.plot(liwc_df.index, liwc_df[col], color=color,
                linewidth=1.5, marker='s', markersize=4, label=label)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('LIWC Rate')
    ax.set_title(title or 'LIWC Category Trend — Raw', pad=10)
    ax.legend(loc='upper left', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


# ── 4-6. LIWC 시계열 (EMA 평활) ──
def plot_liwc_smoothed(liwc_df: pd.DataFrame, title: str = '',
                       smooth_alpha: float = 0.3):
    """EMA 평활 LIWC 시계열"""
    if liwc_df.empty:
        print("⚠️  LIWC 데이터 없음")
        return None

    liwc_cols = {
        'conscientiousness_liwc': ('CON-Achieve', '#00d4ff'),
        'extraversion_liwc':      ('EXT-Social',  '#51cf66'),
        'neuroticism_liwc':       ('NEU-NegEmo',  '#ff6b6b'),
        'agreeableness_liwc':     ('AGR-PosEmo',  '#ffd43b'),
        'openness_liwc':          ('OPN-Insight', '#c084fc'),
    }

    liwc_smoothed = smooth_dataframe(liwc_df, alpha=smooth_alpha)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0a0e1a')

    for col, (label, color) in liwc_cols.items():
        if col not in liwc_df.columns: continue
        # 원본: 반투명 점선
        ax.plot(liwc_df.index, liwc_df[col], color=color,
                linewidth=1, linestyle='--', alpha=0.3, marker='s', markersize=2)
        # EMA: 실선
        ax.plot(liwc_df.index, liwc_smoothed[col], color=color,
                linewidth=1.5, marker='s', markersize=4, label=label)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('LIWC Rate')
    ax.set_title(title or f'LIWC Category Trend — EMA (α={smooth_alpha})', pad=10)
    ax.legend(loc='upper left', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


# ── 4-7. 논문용 종합 테이블 ──
def print_summary_table(metrics: dict, bfi_df: pd.DataFrame):
    buhler = metrics['buhler_delta']
    pr     = metrics['pearson_r']
    dm     = metrics['direction_match']
    base   = bfi_df.iloc[0]
    mid    = bfi_df.iloc[len(bfi_df)//2]
    final  = bfi_df.iloc[-1]

    print("\n" + "="*100)
    print("📋 논문용 종합 결과표")
    print("="*100)
    print(f"{'Dim':>4}  {'Buhler d':>9}  {'Pred':>4}  {'Base':>6}  {'Mid':>6}  {'Final':>6}  {'D Total':>8}  {'Cohen d':>8}  {'Match':>6}  {'Pearson r':>10}")
    print("-"*100)

    for dim in DIMS:
        d       = buhler.get(dim, None)
        pred    = ('↑' if d > 0 else '↓') if d is not None else '—'
        base_v  = f"{base.get(dim, float('nan')):.3f}"
        mid_v   = f"{mid.get(dim, float('nan')):.3f}"
        final_v = f"{final.get(dim, float('nan')):.3f}"
        delta   = metrics['bfi_delta'].get(dim, float('nan'))
        delta_s = f"{delta:+.3f}" if not np.isnan(delta) else '—'
        cohen   = metrics['effect_size'].get(dim, float('nan'))
        cohen_s = f"{cohen:+.4f}" if not np.isnan(cohen) else '—'
        match   = dm.get(dim)
        match_s = '✅' if match else ('❌' if match is False else '—')
        r       = pr.get(dim, float('nan'))
        r_s     = f"{r:+.4f}" if not np.isnan(r) else '—'

        print(f"{DIM_SHORT[dim]:>4}  {str(d) if d else '—':>9}  {pred:>4}  "
              f"{base_v:>6}  {mid_v:>6}  {final_v:>6}  "
              f"{delta_s:>8}  {cohen_s:>8}  {match_s:>4}  {r_s:>10}")

    print("="*100)
    print(f"  Direction Match Rate : {metrics['direction_match_rate']:.1%}")
    if metrics.get('cancel_accuracy') is not None:
        print(f"  Cancel Accuracy      : {metrics['cancel_accuracy']:.1%}")
    print("="*100)


# ── 4-8. 대시보드 공통 하단부 (Δ바차트 / 파이 / Pearson r / LIWC) ──
def _plot_dashboard_bottom(fig, gs, bfi_df, liwc_df, metrics, smooth_alpha, use_smooth):
    """대시보드 하단 3×2 패널 공통 로직"""
    buhler = metrics['buhler_delta']
    pr     = metrics['pearson_r']
    dm     = metrics['direction_match']
    liwc_src = smooth_dataframe(liwc_df, alpha=smooth_alpha) if (use_smooth and not liwc_df.empty) else liwc_df

    # ── 변화량 비교 바차트 ──
    ax2 = fig.add_subplot(gs[1, 0])
    active = [d for d in DIMS if d in buhler]
    x = np.arange(len(active))
    w = 0.35
    ax2.bar(x - w/2, [buhler[d] for d in active], w,
            label='Buhler d', color=[DIM_COLOR[d]+'50' for d in active],
            edgecolor=[DIM_COLOR[d] for d in active], linewidth=1.5)
    ax2.bar(x + w/2, [metrics['bfi_delta'].get(d, 0) for d in active], w,
            label='Actual Δ', color=[DIM_COLOR[d] for d in active], alpha=0.85)
    ax2.axhline(0, color='#64748b', linewidth=0.8, linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels([DIM_SHORT[d] for d in active], fontsize=9)
    ax2.set_title('Expected vs Actual Delta', pad=8)
    ax2.legend(framealpha=0.3, fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # ── 방향 일치 파이 ──
    ax3 = fig.add_subplot(gs[1, 1])
    matched   = sum(1 for v in dm.values() if v == True)
    unmatched = sum(1 for v in dm.values() if v == False)
    pie_vals, pie_labels, pie_colors = [], [], []
    if matched   > 0: pie_vals.append(matched);   pie_labels.append(f'Match ({matched})');    pie_colors.append('#51cf66')
    if unmatched > 0: pie_vals.append(unmatched); pie_labels.append(f'Mismatch ({unmatched})'); pie_colors.append('#ff6b6b')
    if len(pie_vals) >= 2:
        wedges, texts, autotexts = ax3.pie(
            pie_vals, labels=pie_labels, colors=pie_colors,
            autopct='%1.0f%%', startangle=90,
            wedgeprops={'edgecolor': '#0a0e1a', 'linewidth': 2}
        )
        for t in autotexts: t.set_color('#0a0e1a'); t.set_fontweight('bold')
    elif len(pie_vals) == 1:
        color = pie_colors[0]
        ax3.text(0.5, 0.55, pie_labels[0], ha='center', va='center',
                 transform=ax3.transAxes, color=color, fontsize=14, fontweight='bold')
        ax3.text(0.5, 0.38, '100%', ha='center', va='center',
                 transform=ax3.transAxes, color=color, fontsize=22, fontweight='bold')
        ax3.add_patch(plt.Circle((0.5, 0.5), 0.38, color=color, alpha=0.15,
                                 transform=ax3.transAxes))
    else:
        ax3.text(0.5, 0.5, 'No Data', ha='center', va='center',
                 transform=ax3.transAxes, color='#64748b', fontsize=12)
    ax3.set_title(f'Direction Match\n{metrics["direction_match_rate"]:.1%}', pad=8)

    # ── Pearson r ──
    ax4 = fig.add_subplot(gs[1, 2])
    r_dims = [d for d in DIMS if d in pr and not np.isnan(pr[d])]
    r_vals = [pr[d] for d in r_dims]
    r_cols = [DIM_COLOR[d] for d in r_dims]
    bars   = ax4.barh([DIM_SHORT[d] for d in r_dims], r_vals,
                      color=r_cols, alpha=0.85, edgecolor=r_cols, linewidth=1)
    ax4.axvline(0,    color='#64748b', linewidth=0.8, linestyle='--')
    ax4.axvline(0.6,  color='#51cf66', linewidth=1, linestyle=':', alpha=0.5)
    ax4.axvline(-0.6, color='#ff6b6b', linewidth=1, linestyle=':', alpha=0.5)
    for bar, val in zip(bars, r_vals):
        ax4.text(val + (0.03 if val >= 0 else -0.03),
                 bar.get_y() + bar.get_height()/2,
                 f'{val:+.3f}', va='center',
                 ha='left' if val >= 0 else 'right',
                 color='#e2e8f0', fontsize=9)
    ax4.set_xlim(-1.1, 1.1)
    ax4.set_title('Pearson r (BFI vs LIWC)', pad=8)
    ax4.grid(True, alpha=0.3, axis='x')

    # ── LIWC 시계열 ──
    ax5 = fig.add_subplot(gs[2, :])
    if not liwc_df.empty:
        liwc_map = {
            'conscientiousness_liwc': ('#00d4ff', 'CON'),
            'extraversion_liwc':      ('#51cf66', 'EXT'),
            'neuroticism_liwc':       ('#ff6b6b', 'NEU'),
            'agreeableness_liwc':     ('#ffd43b', 'AGR'),
            'openness_liwc':          ('#c084fc', 'OPN'),
        }
        for col, (color, label) in liwc_map.items():
            if col not in liwc_df.columns: continue
            if use_smooth:
                ax5.plot(liwc_df.index, liwc_df[col], color=color,
                         linewidth=1, linestyle='--', alpha=0.25, marker='s', markersize=2)
                ax5.plot(liwc_df.index, liwc_src[col], color=color,
                         linewidth=1.5, marker='s', markersize=3, label=label)
            else:
                ax5.plot(liwc_df.index, liwc_df[col], color=color,
                         linewidth=1.5, marker='s', markersize=3, label=label)
        ax5.set_title('LIWC Category Trend', pad=8)
        ax5.set_ylabel('LIWC Rate')
        ax5.legend(loc='upper left', framealpha=0.3, fontsize=9)
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'No LIWC Data', ha='center', va='center',
                 transform=ax5.transAxes, color='#64748b')
        ax5.set_title('LIWC Timeseries', pad=8)


# ── 4-9. 대시보드 — 원본값 ──
def plot_dashboard_raw(data: dict, metrics: dict, save_path: str = None):
    """논문용 대시보드 — 원본 BFI·LIWC 값"""
    bfi_df, liwc_df = to_dataframe(data)
    buhler  = metrics['buhler_delta']
    events  = data.get('events_detected', [])
    title_str = ' + '.join(e.replace('_',' ').title() for e in events) if events else ''

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0a0e1a')
    fig.suptitle(f'Dynamic Personality Vector — {title_str}  [Raw]',
                 fontsize=16, color='#00d4ff', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # BFI 원본 시계열
    ax1 = fig.add_subplot(gs[0, :])
    all_vals = []
    for dim in DIMS:
        if dim not in bfi_df.columns: continue
        vals  = bfi_df[dim].values
        color = DIM_COLOR[dim]
        ax1.plot(bfi_df.index, vals, color=color, linewidth=2,
                 marker='o', markersize=5, label=DIM_SHORT[dim])
        if dim in buhler:
            d = buhler[dim]
            ax1.annotate(f"{'↑' if d>0 else '↓'}({d:+.2f})",
                         xy=(bfi_df.index[-1], vals[-1]),
                         xytext=(5, 0), textcoords='offset points',
                         color=color, fontsize=8, va='center')
        all_vals.extend(bfi_df[dim].dropna().tolist())

    _set_bfi_yaxis(ax1, all_vals)
    ax1.set_title('BFI Score Trajectory — Raw', pad=8)
    ax1.set_ylabel('BFI Score')
    ax1.legend(loc='upper left', framealpha=0.3, fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    _plot_dashboard_bottom(fig, gs, bfi_df, liwc_df, metrics,
                           smooth_alpha=0.3, use_smooth=False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
        print(f"✅ 저장: {save_path}")
    plt.show()
    return fig


# ── 4-10. 대시보드 — EMA 평활 ──
def plot_dashboard_smoothed(data: dict, metrics: dict, save_path: str = None,
                             smooth_alpha: float = 0.3):
    """논문용 대시보드 — EMA 평활 BFI·LIWC"""
    bfi_df, liwc_df   = to_dataframe(data)
    bfi_smoothed      = smooth_dataframe(bfi_df, alpha=smooth_alpha)
    buhler  = metrics['buhler_delta']
    events  = data.get('events_detected', [])
    title_str = ' + '.join(e.replace('_',' ').title() for e in events) if events else ''

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0a0e1a')
    fig.suptitle(f'Dynamic Personality Vector — {title_str}  [EMA α={smooth_alpha}]',
                 fontsize=16, color='#00d4ff', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # BFI EMA 시계열
    ax1 = fig.add_subplot(gs[0, :])
    all_vals = []
    for dim in DIMS:
        if dim not in bfi_df.columns: continue
        raw_vals    = bfi_df[dim].values
        smooth_vals = bfi_smoothed[dim].values
        color = DIM_COLOR[dim]
        # 원본: 반투명 점선
        ax1.plot(bfi_df.index, raw_vals, color=color, linewidth=1,
                 linestyle='--', alpha=0.25, marker='o', markersize=3)
        # EMA: 실선
        ax1.plot(bfi_df.index, smooth_vals, color=color, linewidth=2,
                 marker='o', markersize=5, label=DIM_SHORT[dim])
        if dim in buhler:
            d = buhler[dim]
            ax1.annotate(f"{'↑' if d>0 else '↓'}({d:+.2f})",
                         xy=(bfi_df.index[-1], smooth_vals[-1]),
                         xytext=(5, 0), textcoords='offset points',
                         color=color, fontsize=8, va='center')
        all_vals.extend(bfi_df[dim].dropna().tolist())

    _set_bfi_yaxis(ax1, all_vals)
    ax1.set_title(f'BFI Score Trajectory — EMA (α={smooth_alpha})', pad=8)
    ax1.set_ylabel('BFI Score')
    ax1.legend(loc='upper left', framealpha=0.3, fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    _plot_dashboard_bottom(fig, gs, bfi_df, liwc_df, metrics,
                           smooth_alpha=smooth_alpha, use_smooth=True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
        print(f"✅ 저장: {save_path}")
    plt.show()
    return fig


# =============================================================================
# 5. 메인 — 전체 실행
# =============================================================================
def plot_all(summary_path: str = None, result_dir: str = None,
             event: str = 'first_job', save_dir: str = None,
             smooth_alpha: float = 0.3):
    """
    전체 분석 실행 — 원본 / EMA 대시보드 각각 생성

    사용법:
      plot_all("results/first_job/summary.json")
      plot_all(result_dir="results", event="first_job")
    """
    if summary_path:
        data = load_summary(summary_path)
    elif result_dir:
        data = load_from_dir(result_dir, event)
    else:
        print("❌ summary_path 또는 result_dir 중 하나를 지정하세요.")
        return

    metrics = compute_metrics(data)
    if not metrics: return

    print_metrics(metrics, data)
    bfi_df, liwc_df = to_dataframe(data)
    print_summary_table(metrics, bfi_df)

    save_raw      = None
    save_smoothed = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ev = '_'.join(data.get('events_detected', [data.get('life_event', 'result')]))
        save_raw      = os.path.join(save_dir, f'{ev}_dashboard_raw.png')
        save_smoothed = os.path.join(save_dir, f'{ev}_dashboard_ema.png')

    # ① 원본값 대시보드
    plot_dashboard_raw(data, metrics, save_path=save_raw)

    # ② EMA 평활 대시보드
    plot_dashboard_smoothed(data, metrics, save_path=save_smoothed,
                            smooth_alpha=smooth_alpha)

    return data, metrics


# =============================================================================
# 주피터에서 바로 실행
# =============================================================================
if __name__ == '__main__':
    SUMMARY_PATH = ""
    SAVE_DIR     = ""

    plot_all(summary_path=SUMMARY_PATH, save_dir=SAVE_DIR)