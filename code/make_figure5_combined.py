# -*- coding: utf-8 -*-
"""Manuscript Figure 5: reconfiguration combined (a: channel slopes, b: annual trajectory).
v2 2026-07-31 — panel (a) colour-harmonised with panel (b): red = strengthening FDR channel,
blue = dissipating FDR channel, grey = non-significant. Data identical to v1."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
OUT='/home/claude/paper4_work/outputs'; FIG='/home/claude/paper4_work/figures'
RED, BLUE = '#c0392b', '#2c6fbb'

rec=json.load(open(f'{OUT}/reconfiguration_result.json')); D=rec['delta']
Z=np.load(f'{OUT}/reconfig_trend.npz'); J=json.load(open(f'{OUT}/reconfig_trend.json'))

fig=plt.figure(figsize=(13.5,6))
gs=fig.add_gridspec(1,2,width_ratios=[1,1.35],wspace=0.28)
axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1])
fig.suptitle('Reconfiguration of role-flow coupling (RQ2)', fontsize=13, y=1.00)

# ---------- (a) channel-level slope chart ----------
order=sorted(D,key=lambda d:d['E_late'])
ys=[d['E_late'] for d in order]; lab_y=list(ys); gap=1.15
for k in range(1,len(lab_y)):
    if lab_y[k]-lab_y[k-1]<gap: lab_y[k]=lab_y[k-1]+gap
for k,d in enumerate(order):
    sig=d['fdr_sig']
    if sig:
        col = RED if d['dE']>0 else BLUE
        lw=2.6; z=4
    else:
        col='0.65'; lw=1.0; z=2
    axA.plot([0,1],[d['E_early'],d['E_late']],'-',color=col,lw=lw,zorder=z)
    axA.plot(0,d['E_early'],'o',color=col,ms=5,zorder=z)
    axA.plot(1,d['E_late'],'o',color=col,ms=5,zorder=z)
    lab=d['pair']+('  *' if sig else '')
    axA.text(1.04,lab_y[k],lab,fontsize=7.8,va='center',color=col,
             fontweight='bold' if sig else 'normal')
axA.axhline(0,color='0.7',lw=0.8)
axA.set_xticks([0,1]); axA.set_xticklabels(['Early\n2008–11','Late\n2022–25'])
axA.set_xlim(-0.15,1.62); axA.set_ylabel('$E_{ij}$ (predicted-direction channel)')
axA.set_title('(a) Channel-level reconfiguration',fontsize=11.5)
axA.spines['bottom'].set_visible(False)

# ---------- (b) annual intensity trajectory ----------
years=Z['years']; inten=Z['inten']; boot=Z['boot_slopes']
xs=years-2008; slope,icpt=np.polyfit(xs,inten,1)
lines=np.array([(inten.mean()+s*(xs-xs.mean())) for s in boot])
band_lo=np.percentile(lines,2.5,axis=0); band_hi=np.percentile(lines,97.5,axis=0)
for (a,b,c) in [(2008,2013,'0.97'),(2014,2019,'0.93'),(2020,2025,'0.97')]:
    axB.axvspan(a-0.5,b+0.5,color=c,zorder=0)
for nm,mid in [('2008–13',2010.5),('2014–19',2016.5),('2020–25',2022.5)]:
    axB.text(mid,4.25,nm,ha='center',fontsize=7.5,color='0.55')
axB.fill_between(years,band_lo,band_hi,color=BLUE,alpha=0.13,lw=0,
                 label='trend 95% band (municipality bootstrap)')
axB.plot(years,inten.mean()+slope*(xs-xs.mean()),'--',color=BLUE,lw=1.6,
         label=f'OLS trend {slope:+.2f}/yr (p={J["annual"]["p"]:.3f})')
axB.plot(years,inten,'-o',color=RED,lw=2,ms=5,label='observed pipeline intensity',zorder=5)
pk=int(years[np.argmax(inten)])
axB.annotate(f'peak {pk} (+{inten.max():.1f})',xy=(pk,inten.max()),
             xytext=(pk-4.6,inten.max()+0.9),fontsize=8.5,color=RED,
             arrowprops=dict(arrowstyle='->',color=RED,lw=1))
axB.set_xticks(range(2008,2026,2)); axB.set_xlim(2007.3,2025.7); axB.set_ylim(4,14)
axB.set_xlabel('year'); axB.set_ylabel('pipeline intensity (mean predicted-direction $E_{ij}$)')
axB.set_title('(b) Annual intensity trajectory, 2008–2025',fontsize=11.5)
axB.legend(loc='upper left',frameon=False,fontsize=8)

def ptxt(p): return 'p<0.001' if p<0.001 else f'p={p:.3f}'
fig.text(0.5,-0.03,
         f"(a) Both periods significant (early intensity {rec['early']['intensity']:+.1f} {ptxt(rec['early']['perm_p'])}; "
         f"late {rec['late']['intensity']:+.1f} {ptxt(rec['late']['perm_p'])}); bold * = FDR-significant channel change "
         f"({rec['fdr_sig']}/13): red = strengthening, blue = dissipating.  "
         "(b) Year-matched roles; linear trend positive and significant; trajectory hump-shaped.",
         ha='center',fontsize=8,style='italic')
plt.tight_layout()
plt.savefig(f'{FIG}/F3_reconfiguration_combined.png',bbox_inches='tight'); plt.close()
print('saved')
