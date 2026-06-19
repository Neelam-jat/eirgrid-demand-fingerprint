import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from analysis import load_demand, fingerprint, old_world_model_test

TEAL='#1F6F6B'; BURG='#8B1E3F'; CREAM='#FAF6EF'; INK='#22303A'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':INK,
    'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK})

h = load_demand()
fp = fingerprint(h)
yrs = fp.index.year

# Chart 1: floor vs peak
fig, ax = plt.subplots(figsize=(9,5.2), dpi=160)
fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
ax.plot(yrs, fp['evening_peak'], color=BURG, lw=2.5, marker='o', ms=5, label='Evening peak (5-7pm avg)')
ax.plot(yrs, fp['night_floor'], color=TEAL, lw=2.5, marker='o', ms=5, label='Night floor (3-5am avg)')
ax.annotate('+21%', xy=(2023, fp['evening_peak'].iloc[-1]), xytext=(2023.1, fp['evening_peak'].iloc[-1]), color=BURG, fontsize=12, fontweight='bold', va='center')
ax.annotate('+39%', xy=(2023, fp['night_floor'].iloc[-1]), xytext=(2023.1, fp['night_floor'].iloc[-1]), color=TEAL, fontsize=12, fontweight='bold', va='center')
ax.set_title("Ireland's night floor is rising almost twice as fast as its peak", fontsize=14, fontweight='bold', pad=14)
ax.set_ylabel('System demand (MW)'); ax.set_xlim(2013.6, 2024.4)
ax.legend(frameon=False, loc='upper left'); ax.grid(axis='y', alpha=0.25)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.text(0.99, 0.01, 'Data: EirGrid 15-min system demand (ROI), 2014-2023', ha='right', fontsize=8, alpha=0.6)
plt.tight_layout(); plt.savefig('charts/01_floor_vs_peak.png', facecolor=CREAM); plt.close()

# Chart 2: model bias
res = old_world_model_test(h)
fig, ax = plt.subplots(figsize=(9,5.2), dpi=160)
fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
bars = ax.bar([str(y) for y in res.index], res['ml_bias'], color=[TEAL]+[BURG]*3, width=0.55)
for b, v in zip(bars, res['ml_bias']):
    ax.text(b.get_x()+b.get_width()/2, v+1.5, f'+{v:.0f} MW', ha='center', fontweight='bold')
ax.set_title('A model trained on 2015-2019 under-forecasts every later year,\nand the miss grows', fontsize=14, fontweight='bold', pad=14)
ax.set_ylabel('Average under-forecast (MW)')
ax.grid(axis='y', alpha=0.25)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.text(0.99, 0.01, 'Gradient boosting, calendar split, hourly EirGrid ROI demand', ha='right', fontsize=8, alpha=0.6)
plt.tight_layout(); plt.savefig('charts/02_model_bias.png', facecolor=CREAM); plt.close()
print("charts saved")
