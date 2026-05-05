"""
dataset_analysis.py
--------------------
Complete dataset analysis + balancing for IoT-23 cleaned data.

Datasets used:
    Minor_datset/IoT23/cleaned_data.csv     (~48k rows)
    Minor_datset/IoT23/iot23_combined.csv   (~1.4M rows)

Run:
    python3 dataset_analysis.py

Outputs (in analysis_output/):
    01_raw_class_distribution.png
    02_per_file_distribution.png
    03_feature_correlation_heatmap.png
    04_feature_distributions.png
    05_balanced_class_distribution.png
    06_pca_visualization.png
    07_attack_type_breakdown.png
    dataset_report.txt
    balanced_dataset.csv   <- use THIS for training
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_FILES = [
    'Minor_datset/IoT23/cleaned_data.csv',
    'Minor_datset/IoT23/iot23_combined.csv',
]
LABEL_COL    = 'label'
NORMAL_LABEL = 'Benign'
OUTPUT_DIR   = 'analysis_output'
BALANCE_CAP  = 150_000
CHUNK_SIZE   = 100_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────

C = {
    'normal': '#10b981', 'attack': '#ef4444',
    'bg':     '#111827', 'text':   '#e2e8f0',
    'grid':   '#2a3347', 'accent': '#00d4ff',
}
plt.rcParams.update({
    'figure.facecolor': C['bg'], 'axes.facecolor': C['bg'],
    'axes.edgecolor':   C['grid'], 'axes.labelcolor': C['text'],
    'xtick.color': C['text'],     'ytick.color': C['text'],
    'text.color':  C['text'],     'grid.color':  C['grid'],
    'grid.alpha':  0.4,           'font.size':   10,
})

def save(fig, name):
    path = f'{OUTPUT_DIR}/{name}'
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"  Saved -> {path}")


# =============================================================================
# STEP 1 — Load data
# =============================================================================
print("\n" + "="*60)
print("  STEP 1 — Loading datasets")
print("="*60)

all_dfs    = []
file_stats = []

for path in DATASET_FILES:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        continue

    print(f"\n  Loading: {path}")
    chunks, rows = [], 0
    for chunk in pd.read_csv(path, low_memory=False, chunksize=CHUNK_SIZE):
        chunks.append(chunk)
        rows += len(chunk)
        print(f"    ... {rows:,} rows", end='\r')

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n  Loaded: {len(df):,} rows, {len(df.columns)} columns")

    if LABEL_COL not in df.columns:
        print(f"  [warn] No '{LABEL_COL}' column — skipping")
        continue

    df['label_orig'] = df[LABEL_COL].astype(str).str.strip()
    df[LABEL_COL]    = df['label_orig'].apply(
        lambda x: 0 if x == NORMAL_LABEL else 1)

    n_normal = (df[LABEL_COL] == 0).sum()
    n_attack = (df[LABEL_COL] == 1).sum()
    pct      = 100 * n_attack / len(df)

    file_stats.append({
        'file': os.path.basename(path), 'total': len(df),
        'normal': n_normal, 'attack': n_attack, 'attack_pct': round(pct, 1)
    })
    print(f"  Normal: {n_normal:,}  |  Attack: {n_attack:,}  |  Attack%: {pct:.1f}%")
    all_dfs.append(df)

if not all_dfs:
    raise RuntimeError("No datasets loaded — check paths.")

df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
df_all = df_all.drop_duplicates()
df_all = df_all.dropna(subset=[LABEL_COL])
df_all[LABEL_COL] = df_all[LABEL_COL].astype(int)

total_normal = (df_all[LABEL_COL] == 0).sum()
total_attack = (df_all[LABEL_COL] == 1).sum()
total        = len(df_all)
ratio        = total_attack / max(total_normal, 1)

print(f"\n  Combined (after dedup): {total:,} rows")
print(f"  Normal : {total_normal:,}  ({100*total_normal/total:.1f}%)")
print(f"  Attack : {total_attack:,}  ({100*total_attack/total:.1f}%)")
print(f"  Imbalance ratio: {ratio:.1f}x")

# ── Feature list ──────────────────────────────────────────────────────────────

drop_always = [LABEL_COL, 'label_orig']
feat_cols   = [c for c in df_all.columns
               if c not in drop_always
               and df_all[c].dtype in ['float64','float32','int64','int32','uint8']]

print(f"\n  Numeric features: {len(feat_cols)}")
print(f"  Features: {feat_cols}")


# =============================================================================
# PLOT 1 — Raw class distribution
# =============================================================================
print("\n" + "="*60)
print("  PLOT 1 — Raw class distribution")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

bars = ax1.bar(['Normal', 'Attack'], [total_normal, total_attack],
               color=[C['normal'], C['attack']], width=0.45, edgecolor='none')
ax1.set_title('Raw Sample Counts', fontsize=12, fontweight='bold', pad=10)
ax1.set_ylabel('Number of Samples')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x):,}'))
for bar, val in zip(bars, [total_normal, total_attack]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
             f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.grid(axis='y')

ax2.pie([total_normal, total_attack],
        labels=[f'Normal\n{total_normal:,}', f'Attack\n{total_attack:,}'],
        colors=[C['normal'], C['attack']], startangle=90,
        wedgeprops=dict(edgecolor='#0a0e1a', linewidth=2),
        autopct='%1.1f%%', pctdistance=0.78,
        textprops={'fontsize': 11})
ax2.set_title('Class Proportion', fontsize=12, fontweight='bold', pad=10)

fig.suptitle('IoT-23 Dataset — Raw Class Distribution',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, '01_raw_class_distribution.png')


# =============================================================================
# PLOT 2 — Per-file distribution
# =============================================================================
print("\n" + "="*60)
print("  PLOT 2 — Per-file distribution")
print("="*60)

fig, axes = plt.subplots(1, len(file_stats), figsize=(6*len(file_stats), 5))
if len(file_stats) == 1:
    axes = [axes]

for ax, s in zip(axes, file_stats):
    bars = ax.bar(['Normal', 'Attack'], [s['normal'], s['attack']],
                  color=[C['normal'], C['attack']], width=0.45, edgecolor='none')
    ax.set_title(s['file'], fontsize=10, fontweight='bold', pad=8)
    ax.set_ylabel('Sample Count')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax.grid(axis='y')
    for bar, val in zip(bars, [s['normal'], s['attack']]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f'{val:,}', ha='center', va='bottom', fontsize=10)
    ax.text(0.5, 0.96, f"Attack: {s['attack_pct']}%",
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            color=C['attack'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1c2333',
                      edgecolor=C['attack'], alpha=0.85))

fig.suptitle('Class Distribution Per Dataset File',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, '02_per_file_distribution.png')


# =============================================================================
# PLOT 3 — Attack type breakdown
# =============================================================================
print("\n" + "="*60)
print("  PLOT 3 — Attack type breakdown")
print("="*60)

attack_counts = (df_all[df_all[LABEL_COL] == 1]['label_orig']
                 .value_counts().head(15))

fig, ax = plt.subplots(figsize=(11, 6))
colors_bar = plt.cm.RdYlGn_r(np.linspace(0.1, 0.85, len(attack_counts)))
bars = ax.barh(attack_counts.index[::-1], attack_counts.values[::-1],
               color=colors_bar[::-1], edgecolor='none')

for bar, val in zip(bars, attack_counts.values[::-1]):
    ax.text(bar.get_width() + attack_counts.max()*0.005,
            bar.get_y()+bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)

ax.set_xlabel('Number of Samples', fontsize=11)
ax.set_title('Attack Type Distribution (Top 15)',
             fontsize=13, fontweight='bold', pad=12)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x):,}'))
ax.grid(axis='x')
plt.tight_layout()
save(fig, '07_attack_type_breakdown.png')


# =============================================================================
# PLOT 4 — Feature distributions
# =============================================================================
print("\n" + "="*60)
print("  PLOT 4 — Feature distributions")
print("="*60)

normal_df = df_all[df_all[LABEL_COL] == 0]
attack_df = df_all[df_all[LABEL_COL] == 1]
plot_feats = feat_cols[:16]
n_cols = 4
n_rows = (len(plot_feats) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*3))
axes = axes.flatten()

for i, feat in enumerate(plot_feats):
    ax = axes[i]
    try:
        lo = df_all[feat].quantile(0.01)
        hi = df_all[feat].quantile(0.99)
        ax.hist(normal_df[feat].dropna().clip(lo, hi), bins=50,
                alpha=0.65, color=C['normal'], label='Normal', density=True)
        ax.hist(attack_df[feat].dropna().clip(lo, hi), bins=50,
                alpha=0.65, color=C['attack'], label='Attack', density=True)
        ax.set_title(feat, fontsize=9, fontweight='bold', pad=4)
        ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis='y')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=7)

for j in range(len(plot_feats), len(axes)):
    axes[j].set_visible(False)

handles = [plt.Rectangle((0,0),1,1, color=C['normal'], alpha=0.7),
           plt.Rectangle((0,0),1,1, color=C['attack'],  alpha=0.7)]
fig.legend(handles, ['Normal', 'Attack'], loc='lower right',
           fontsize=11, framealpha=0.3)
fig.suptitle('Feature Distributions — Normal vs Attack Traffic',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '04_feature_distributions.png')


# =============================================================================
# PLOT 5 — Correlation heatmap
# =============================================================================
print("\n" + "="*60)
print("  PLOT 5 — Correlation heatmap")
print("="*60)

sample_corr = df_all[feat_cols].sample(min(20000, len(df_all)), random_state=42)
corr = sample_corr.corr()

fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, ax=ax,
            linewidths=0.4, linecolor='#0a0e1a',
            cbar_kws={'shrink': 0.8, 'label': 'Correlation coefficient'},
            annot=len(feat_cols) <= 15, fmt='.2f', annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix',
             fontsize=13, fontweight='bold', pad=12)
ax.tick_params(labelsize=8)
plt.tight_layout()
save(fig, '03_feature_correlation_heatmap.png')


# =============================================================================
# PLOT 6 — PCA 2D visualization
# =============================================================================
print("\n" + "="*60)
print("  PLOT 6 — PCA 2D visualization")
print("="*60)

sample_size = min(8000, len(df_all))
df_pca = df_all.sample(sample_size, random_state=42)
X_pca  = df_pca[feat_cols].fillna(0).values
y_pca  = df_pca[LABEL_COL].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_pca)
pca    = PCA(n_components=2, random_state=42)
X_2d   = pca.fit_transform(X_sc)

fig, ax = plt.subplots(figsize=(10, 7))
for lbl, col, name in [(0, C['normal'], 'Normal'), (1, C['attack'], 'Attack')]:
    m = y_pca == lbl
    ax.scatter(X_2d[m,0], X_2d[m,1], c=col, alpha=0.35, s=10,
               label=f'{name} ({m.sum():,})', edgecolors='none')

ax.set_xlabel(f'PC1 — {pca.explained_variance_ratio_[0]*100:.1f}% variance',
              fontsize=11)
ax.set_ylabel(f'PC2 — {pca.explained_variance_ratio_[1]*100:.1f}% variance',
              fontsize=11)
ax.set_title('PCA 2D Projection — Normal vs Attack Traffic',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11, markerscale=3, framealpha=0.3)
ax.grid(True)
plt.tight_layout()
save(fig, '06_pca_visualization.png')


# =============================================================================
# STEP 3 — Balance dataset
# =============================================================================
print("\n" + "="*60)
print("  STEP 3 — Balancing dataset")
print("="*60)

n_df   = df_all[df_all[LABEL_COL] == 0]
a_df   = df_all[df_all[LABEL_COL] == 1]
target = min(len(n_df), len(a_df), BALANCE_CAP)

print(f"  Normal available : {len(n_df):,}")
print(f"  Attack available : {len(a_df):,}")
print(f"  Target per class : {target:,}")

n_sampled = n_df.sample(target, random_state=42)
a_sampled = a_df.sample(target, random_state=42)

df_bal = pd.concat([n_sampled, a_sampled], ignore_index=True)
df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)
df_bal = df_bal.drop(columns=['label_orig'], errors='ignore')

bal_n = (df_bal[LABEL_COL] == 0).sum()
bal_a = (df_bal[LABEL_COL] == 1).sum()

print(f"  Balanced Normal : {bal_n:,}")
print(f"  Balanced Attack : {bal_a:,}")
print(f"  Total           : {len(df_bal):,}")

bal_path = f'{OUTPUT_DIR}/balanced_dataset.csv'
df_bal.to_csv(bal_path, index=False)
print(f"  Saved -> {bal_path}")

# Balanced distribution plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

bars = ax1.bar(['Normal', 'Attack'], [bal_n, bal_a],
               color=[C['normal'], C['attack']], width=0.45, edgecolor='none')
ax1.set_title('Balanced Sample Counts', fontsize=12, fontweight='bold', pad=10)
ax1.set_ylabel('Number of Samples')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x):,}'))
ax1.grid(axis='y')
for bar, val in zip(bars, [bal_n, bal_a]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
             f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.pie([bal_n, bal_a],
        labels=[f'Normal\n{bal_n:,}', f'Attack\n{bal_a:,}'],
        colors=[C['normal'], C['attack']], startangle=90,
        wedgeprops=dict(edgecolor='#0a0e1a', linewidth=2),
        autopct='%1.1f%%', pctdistance=0.75,
        textprops={'fontsize': 11})
ax2.set_title('Balanced Class Proportion', fontsize=12, fontweight='bold', pad=10)

fig.suptitle('After Balancing — Perfect 50/50 Split',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, '05_balanced_class_distribution.png')


# =============================================================================
# STEP 4 — Write text report
# =============================================================================
print("\n" + "="*60)
print("  STEP 4 — Writing dataset report")
print("="*60)

df_feat  = df_all[feat_cols]
f_stats  = pd.DataFrame({
    'mean':     df_feat.mean(),
    'std':      df_feat.std(),
    'min':      df_feat.min(),
    'max':      df_feat.max(),
    'missing%': (df_feat.isnull().sum() / len(df_feat) * 100).round(2),
    'variance': df_feat.var(),
}).sort_values('variance', ascending=False)

attack_type_counts = (df_all[df_all[LABEL_COL]==1]['label_orig']
                      .value_counts())

lines = [
    "=" * 65,
    "  DATASET ANALYSIS REPORT",
    "  Project : XAI-Powered Secure IoT Firewall",
    "  Dataset : IoT-23 (Cleaned)",
    "=" * 65, "",
    "1. DATASET FILES", "-"*40,
]
for s in file_stats:
    lines += [
        f"  File       : {s['file']}",
        f"  Total rows : {s['total']:,}",
        f"  Normal     : {s['normal']:,}",
        f"  Attack     : {s['attack']:,}",
        f"  Attack %   : {s['attack_pct']}%", "",
    ]

lines += [
    "2. COMBINED RAW DATASET", "-"*40,
    f"  Total rows      : {total:,}",
    f"  Total features  : {len(feat_cols)}",
    f"  Normal samples  : {total_normal:,}  ({100*total_normal/total:.2f}%)",
    f"  Attack samples  : {total_attack:,}  ({100*total_attack/total:.2f}%)",
    f"  Imbalance ratio : {ratio:.2f}x  (attack:normal)", "",
    "3. ATTACK TYPE BREAKDOWN", "-"*40,
]
for atype, cnt in attack_type_counts.items():
    lines.append(f"  {atype:<45} {cnt:>8,}  ({100*cnt/total_attack:.1f}%)")

lines += [
    "", "4. BALANCING STRATEGY", "-"*40,
    "  Method          : Random undersampling of majority class",
    "  Justification   : Preserves real data, no synthetic samples —",
    "                    safer and more honest for security research",
    f"  Target per class: {target:,}",
    f"  Final Normal    : {bal_n:,}",
    f"  Final Attack    : {bal_a:,}",
    f"  Final Total     : {len(df_bal):,}",
    "  Balance ratio   : 1:1  (50% normal / 50% attack)", "",
    "5. FEATURE STATISTICS", "-"*40,
    f"  {'Feature':<35} {'Mean':>10} {'Std':>10} "
    f"{'Min':>10} {'Max':>10} {'Missing%':>9}",
    "-"*90,
]
for feat, row in f_stats.iterrows():
    lines.append(
        f"  {feat:<35} {row['mean']:>10.3f} {row['std']:>10.3f} "
        f"{row['min']:>10.3f} {row['max']:>10.3f} {row['missing%']:>8.2f}%"
    )

lines += [
    "", "6. PCA VARIANCE EXPLAINED", "-"*40,
    f"  PC1  : {pca.explained_variance_ratio_[0]*100:.2f}%",
    f"  PC2  : {pca.explained_variance_ratio_[1]*100:.2f}%",
    f"  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%  (2 components)",
    "", "7. OUTPUT FILES", "-"*40,
    "  01_raw_class_distribution.png      Raw Normal vs Attack (bar + pie)",
    "  02_per_file_distribution.png       Class split per input file",
    "  03_feature_correlation_heatmap.png Correlation between all features",
    "  04_feature_distributions.png       Feature histograms: Normal vs Attack",
    "  05_balanced_class_distribution.png Post-balancing split (50/50)",
    "  06_pca_visualization.png           2D PCA scatter of traffic samples",
    "  07_attack_type_breakdown.png       Count of each attack type",
    "  balanced_dataset.csv               Ready-to-train balanced dataset",
    "",
    "=" * 65,
    "  CONCLUSION & RECOMMENDATION FOR RESEARCH MENTOR",
    "=" * 65,
    f"  The raw IoT-23 dataset contains {total:,} samples with a significant",
    f"  class imbalance ({ratio:.1f}x more attacks than normal traffic).",
    "  Training on this raw data would bias the model towards always",
    "  predicting attacks, reducing its practical usefulness.",
    "",
    "  After applying random undersampling, the dataset is balanced",
    f"  at {target:,} samples per class, giving a perfect 1:1 ratio.",
    "  This ensures the CNN-BiLSTM model learns both normal and",
    "  attack traffic patterns equally well.",
    "",
    "  The balanced_dataset.csv is recommended for model training.",
    "=" * 65,
]

report_path = f'{OUTPUT_DIR}/dataset_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(lines))
print(f"  Saved -> {report_path}")


# =============================================================================
# Done
# =============================================================================
print("\n" + "="*60)
print("  ALL DONE")
print("="*60)
print(f"""
  Raw       ->  {total:,} rows
                Normal : {total_normal:,}  ({100*total_normal/total:.1f}%)
                Attack : {total_attack:,}  ({100*total_attack/total:.1f}%)
                Ratio  : {ratio:.1f}x imbalance

  Balanced  ->  {len(df_bal):,} rows  (50/50)
                Normal : {bal_n:,}
                Attack : {bal_a:,}

  Features  ->  {len(feat_cols)} numeric features
  Plots     ->  7 charts in {OUTPUT_DIR}/
  Report    ->  {OUTPUT_DIR}/dataset_report.txt
  Training  ->  {OUTPUT_DIR}/balanced_dataset.csv
""")