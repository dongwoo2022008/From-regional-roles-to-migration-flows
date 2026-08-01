"""F5: 에스컬레이터 서명 — (a) ESC 총 순이동 by age (청년유입/가족기유출), (b) 공급측 3채널 feed→return 반전"""
import numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
OUT='/home/claude/paper4_work/outputs'; FIG='/home/claude/paper4_work/figures'
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']
Z=np.load(f'{OUT}/tensor_static.npz',allow_pickle=True); F=Z['F']
AGELAB=['≤10','10s','20s','30s','40s','50s','60s','70+']; ei=ROLES.index('ESC')
# (a) ESC net by age
net=[]
for ai in range(8):
    Fa=F[:,ai].sum(axis=0); net.append((Fa[:,ei].sum()-Fa[ei,ei])-(Fa[ei,:].sum()-Fa[ei,ei]))
net=np.array(net)/1000.0
# (b) source->ESC E by age
def E_age(i,j):
    ii,jj=ROLES.index(i),ROLES.index(j); o=[]
    for ai in range(8):
        Fa=F[:,ai].sum(axis=0); T=Fa[ii,jj]+Fa[jj,ii]; o.append(100*(Fa[ii,jj]-Fa[jj,ii])/T if T>0 else np.nan)
    return o

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
# panel a
cols=['#c0392b' if v>0 else '#2c6fbb' for v in net]
ax1.bar(range(8),net,color=cols,width=0.7)
ax1.axhline(0,color='0.4',lw=0.9)
for k,v in enumerate(net):
    ax1.text(k, v+(15 if v>0 else -15), f'{v:+.0f}k', ha='center',va='bottom' if v>0 else 'top',fontsize=8,
             color='#c0392b' if v>0 else '#2c6fbb')
ax1.set_xticks(range(8)); ax1.set_xticklabels(AGELAB)
ax1.set_ylabel('net internal migration of escalator regions (×1000)')
ax1.set_xlabel('age group'); ax1.set_title('(a) Escalator regions gain the young, shed the family-stage',fontsize=11)
ax1.set_ylim(-560,560)
ax1.annotate('accumulation\n(20s, +459k)',xy=(2,459),xytext=(2.4,300),fontsize=8,color='#c0392b',arrowprops=dict(arrowstyle='->',color='#c0392b',lw=1))
ax1.annotate('dispersal\n(30s, −483k)',xy=(3,-483),xytext=(3.4,-330),fontsize=8,color='#2c6fbb',arrowprops=dict(arrowstyle='->',color='#2c6fbb',lw=1))
# panel b
for src,col in [('SUP','#c0392b'),('ANC','#e08214'),('OUT','#6a51a3')]:
    v=E_age(src,'ESC'); ax2.plot(range(8),v,'-o',color=col,lw=2,ms=5,label=f'{src}→ESC')
    ax2.text(7.05,v[-1],src,color=col,fontsize=8,va='center')
ax2.axhline(0,color='0.4',lw=0.9)
ax2.axvspan(1.5,2.5,color='0.92',zorder=0)
ax2.text(2,32,'feed\n(young in)',ha='center',fontsize=7.5,color='0.4')
ax2.text(5.5,-24,'return / counterstream\n(older back out)',ha='center',fontsize=7.5,color='0.4')
ax2.set_xticks(range(8)); ax2.set_xticklabels(AGELAB); ax2.set_xlim(-0.3,7.6); ax2.set_ylim(-30,38)
ax2.set_xlabel('age group'); ax2.set_ylabel('$E$(source→ESC)  (+: net into escalator)')
ax2.set_title('(b) Source→escalator flows reverse after the 20s',fontsize=11)
ax2.legend(loc='lower left',frameon=False,fontsize=8)
fig.suptitle('The escalator signature in directional role flows (2006–2025)',fontsize=12.5,y=1.01)
fig.text(0.5,-0.02,'Roles measured in Paper ③ from marginals only; here the escalator role is shown to function directionally — net inflow of 20-somethings, net outflow at family-formation and later ages. Association, not causation.',
         ha='center',fontsize=8,style='italic')
plt.tight_layout(); plt.savefig(f'{FIG}/F5_escalator_signature.png',bbox_inches='tight'); plt.close()
print('F5 saved')
