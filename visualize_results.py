"""Visualize training results for Baseline / KAN Head / KAN Hybrid."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 5),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ── Data parsed from training logs ──

baseline_val = [
    0.82191, 0.83589, 0.84167, 0.83944, 0.84083, 0.84562, 0.84416, 0.84655,
    0.84798, 0.84889, 0.84834, 0.84889, 0.84764, 0.84999, 0.84965, 0.84883,
    0.85039, 0.84960, 0.84872, 0.84895, 0.84941, 0.84889, 0.84979, 0.84839,
    0.84912, 0.84935, 0.84886, 0.84960, 0.84932, 0.84899, 0.84860, 0.84943,
    0.84750, 0.84937, 0.84890, 0.84875, 0.84878, 0.84980, 0.84957, 0.84918,
    0.85009, 0.84982, 0.84935, 0.84950, 0.85027, 0.84962, 0.84971, 0.84994,
    0.85034, 0.85016,
]

kan_head_val = [
    0.82396, 0.83680, 0.84132, 0.84290, 0.84147, 0.84538, 0.84593, 0.84334,
    0.84854, 0.84983, 0.84827, 0.84418, 0.84872, 0.84871, 0.84817, 0.84802,
    0.85036, 0.85052, 0.84906, 0.84916, 0.84944, 0.84934, 0.84921, 0.84993,
    0.84834, 0.84920, 0.84727, 0.84948, 0.84956, 0.84919, 0.84817, 0.84844,
    0.84846, 0.84914, 0.84830, 0.84885, 0.84951, 0.84988, 0.84923, 0.84978,
    0.85005, 0.85016, 0.84957, 0.85025, 0.85021, 0.85009, 0.84934, 0.85037,
    0.85096, 0.84998,
]

kan_hybrid_val = [
    0.82152, 0.83572, 0.83914, 0.84121, 0.83845, 0.84404, 0.84491, 0.84464,
    0.84494, 0.84684, 0.84661, 0.84587, 0.84509, 0.84632, 0.84594, 0.84674,
    0.84877, 0.84770, 0.84701, 0.84797, 0.84657,
    0.10000, 0.09996, 0.10002, 0.10001, 0.10001, 0.09999, 0.09999, 0.10001,
    0.09997, 0.10001, 0.09999, 0.09999, 0.10001, 0.10002, 0.10000, 0.09993,
    0.10002, 0.10002, 0.10003, 0.09996, 0.10002, 0.10005, 0.09994, 0.10001,
    0.09998, 0.10004, 0.10001, 0.09999, 0.09997,
]

epochs = np.arange(50)

# ── Figure 1: Validation Accuracy Curves ──

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, baseline_val, 'o-', markersize=3, label=f'Baseline (best={max(baseline_val):.5f})', color='#2196F3')
ax1.plot(epochs, kan_head_val, 's-', markersize=3, label=f'KAN Head v1 (best={max(kan_head_val):.5f})', color='#4CAF50')
ax1.plot(epochs, kan_hybrid_val, '^-', markersize=3, label=f'KAN Hybrid v2 (best={max(kan_hybrid_val[:21]):.5f}*)', color='#F44336')
ax1.axvline(x=21, color='#F44336', linestyle='--', alpha=0.5, label='v2 collapse (epoch 21)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Accuracy')
ax1.set_title('Validation Accuracy over Training')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_ylim([0.0, 0.86])

# zoomed-in view (only stable region)
ax2.plot(epochs, baseline_val, 'o-', markersize=3, label='Baseline', color='#2196F3')
ax2.plot(epochs, kan_head_val, 's-', markersize=3, label='KAN Head v1', color='#4CAF50')
ax2.plot(epochs[:21], kan_hybrid_val[:21], '^-', markersize=3, label='KAN Hybrid v2 (before collapse)', color='#F44336')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy')
ax2.set_title('Zoomed: Stable Training Region')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_ylim([0.82, 0.855])

plt.tight_layout()
plt.savefig('validation_accuracy_curves.png', bbox_inches='tight')
plt.close()
print('Saved: validation_accuracy_curves.png')

# ── Figure 2: Best Accuracy Comparison Bar Chart ──

fig, ax = plt.subplots(figsize=(8, 5))

models = ['Baseline\n(ParT)', 'KAN Head\n(v1)', 'KAN Hybrid\n(v2)']
best_accs = [max(baseline_val), max(kan_head_val), max(kan_hybrid_val[:21])]
colors = ['#2196F3', '#4CAF50', '#F44336']
best_epochs = [
    np.argmax(baseline_val),
    np.argmax(kan_head_val),
    np.argmax(kan_hybrid_val[:21]),
]

bars = ax.bar(models, best_accs, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

for bar, acc, ep in zip(bars, best_accs, best_epochs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
            f'{acc:.5f}\n(epoch {ep})', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Best Validation Accuracy')
ax.set_title('Best Validation Accuracy Comparison')
ax.set_ylim([0.845, 0.853])
ax.axhline(y=max(baseline_val), color='#2196F3', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('best_accuracy_comparison.png', bbox_inches='tight')
plt.close()
print('Saved: best_accuracy_comparison.png')

# ── Summary ──

print('\n' + '=' * 60)
print('RESULTS SUMMARY')
print('=' * 60)
print(f'{"Model":<20} {"Best Val Acc":<15} {"Best Epoch":<12} {"Status"}')
print('-' * 60)
print(f'{"Baseline (ParT)":<20} {max(baseline_val):<15.5f} {np.argmax(baseline_val):<12} Normal')
print(f'{"KAN Head (v1)":<20} {max(kan_head_val):<15.5f} {np.argmax(kan_head_val):<12} Normal')
print(f'{"KAN Hybrid (v2)":<20} {max(kan_hybrid_val[:21]):<15.5f} {np.argmax(kan_hybrid_val[:21]):<12} Collapsed@21')
print('-' * 60)
print(f'v1 vs Baseline: {max(kan_head_val) - max(baseline_val):+.5f} ({"better" if max(kan_head_val) > max(baseline_val) else "worse"})')
print(f'v2 vs Baseline: {max(kan_hybrid_val[:21]) - max(baseline_val):+.5f} (before collapse)')
print('=' * 60)
