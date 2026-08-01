"""F4: annual pipeline intensity trajectory 2008-2025 + OLS trend line + bootstrap band"""
import json,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
OUT='/home/claude/paper4_work/outputs'; FIG='/home/claude/paper4_work/figures'
Z=np.load(f'{OUT}/reconfig_trend.npz'); J=json.load(open(f'{OUT}/reconfig_trend.json'))
years=Z['years']; inten=Z['inten']; boot=Z['boot_slopes']
xs=years-2008; slope,icpt=np.polyfit(xs,inten,1)
# Bootstrap prediction band (slope distribution -> distribution of trend lines per year; intercept refit around the mean curve)
rng=np.random.default_rng(20260716)
# Band: 5-95% of the trend lines drawn from the bootstrap slopes at each year
lines=np.array([ (inten.mean()+s*(xs-xs.mean())) for s in boot])
band_lo=np.percentile(lines,2.5,axis=0); band_hi=np.percentile(lines,97.5,axis=0)

fig,ax=plt.subplots(figsize=(8.4,5))
ax.axhline(0,color='0.6',lw=0.8)
ax.fill_between(years,band_lo,band_hi,color='#2c6fbb',alpha=0.13,lw=0,label='trend 95% band (region bootstrap)')
ax.plot(years,inten.mean()+slope*(xs-xs.mean()),'--',color='#2c6fbb',lw=1.6,label=f'OLS trend {slope:+.2f}/yr (p={J["annual"]["p"]:.3f})')
ax.plot(years,inten,'-o',color='#c0392b',lw=2,ms=5,label='observed pipeline intensity',zorder=5)
pk=int(years[np.argmax(inten)])
ax.annotate(f'peak {pk}\n(+{inten.max():.1f})',xy=(pk,inten.max()),xytext=(pk-3.2,inten.max()+1.2),
            fontsize=8.5,color='#c0392b',arrowprops=dict(arrowstyle='->',color='#c0392b',lw=1))
# Three-period background shading
for (a,b,c) in [(2008,2013,'0.97'),(2014,2019,'0.93'),(2020,2025,'0.97')]:
    ax.axvspan(a-0.5,b+0.5,color=c,zorder=0)
for nm,mid in [('2008–13',2010.5),('2014–19',2016.5),('2020–25',2022.5)]:
    ax.text(mid,4.25,nm,ha='center',fontsize=7.5,color='0.55')
ax.set_xticks(range(2008,2026,2)); ax.set_xlim(2007.3,2025.7); ax.set_ylim(4,14)
ax.set_xlabel('year'); ax.set_ylabel('pipeline intensity\n(mean predicted-direction $E_{ij}$, 13 channels)')
ax.set_title('Trajectory of role-flow pipeline intensity, 2008–2025',fontsize=12)
ax.legend(loc='upper left',frameon=False,fontsize=8)
fig.text(0.5,-0.02,'Annual role-flow coupling (year-matched roles). Linear trend significantly positive; trajectory hump-shaped — intensifies through the 2010s, peaks in the late 2010s, partially recedes after 2020 while staying above 2008.',
         ha='center',fontsize=7.8,style='italic',wrap=True)
plt.tight_layout(); plt.savefig(f'{FIG}/F4_intensity_trajectory.png',bbox_inches='tight'); plt.close()
print('F4 saved')
