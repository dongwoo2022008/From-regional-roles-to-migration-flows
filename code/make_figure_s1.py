"""FS1 (supplementary): permutation null distribution of sign matches vs observed 11/13."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

null = np.load('/home/claude/paper4_work/outputs/permutation_null.npz')['null_match']
obs = 11

fig, ax = plt.subplots(figsize=(7.2, 4.2))
bins = np.arange(-0.5, 14.5, 1)
ax.hist(null, bins=bins, color='#9db8d6', edgecolor='white', density=True, label='null (B = 1000 label permutations)')
ax.axvline(obs, color='#c0392b', lw=2.2)
ax.annotate(f'observed = {obs}\np = {(null >= obs).mean():.3f}', xy=(obs, ax.get_ylim()[1]*0.75),
            xytext=(obs-4.6, ax.get_ylim()[1]*0.78), color='#c0392b', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#c0392b'))
ax.set_xlabel('number of pre-specified signs recovered (of 13)')
ax.set_ylabel('density')
ax.set_xticks(range(0, 14))
ax.set_title('Permutation null distribution of the global sign-match statistic', fontsize=11.5)
ax.legend(frameon=False, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
txt = f'null mean {null.mean():.2f}, 95% interval [{int(np.percentile(null,2.5))}, {int(np.percentile(null,97.5))}]'
ax.text(0.02, 0.97, txt, transform=ax.transAxes, va='top', fontsize=9, color='0.4')
plt.tight_layout()
plt.savefig('/home/claude/paper4_work/figures/FS1_permutation_null.png', dpi=200, bbox_inches='tight')
print('FS1 saved')
