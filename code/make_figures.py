"""논문④ Step7: 핵심 figure 3종 (저널용, 영문 라벨)
F1 파이프라인 부호 히트맵(예측 vs 관측 E_ij + FDR 유의)
F2 연령정합 곡선(생애과정 서명): SUP→ESC, ESC→LAND, ESC→HTR by age
F3 재편 슬로프: E_early -> E_late (확증 13쌍), FDR 유의 강조
"""
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':0.8,
                     'figure.dpi':150,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
OUT='/home/claude/paper4_work/outputs'; FIG='/home/claude/paper4_work/figures'
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']
RLAB=['Sup-ret\n(SUP)','Anchor\n(ANC)','Grad-out\n(OUT)','Escal.\n(ESC)','Landing\n(LAND)','Hi-turn\n(HTR)']
DIV='RdBu_r'   # 발산형: 파랑(순유입<0) - 회백(0) - 빨강(순유출>0)

# ---------- F1 ----------
eff=np.load(f'{OUT}/effectiveness_static.npz',allow_pickle=True); E=eff['E_all']
perm=json.load(open(f'{OUT}/permutation_result.json'))
fdrsig={c['pair']:c['fdr_sig'] for c in perm['cells']}
# 예측 부호행렬
M={'SUP':{'ANC':-1,'OUT':-1,'ESC':1,'LAND':1,'HTR':1},'ANC':{'SUP':1,'OUT':0,'ESC':1,'LAND':1,'HTR':1},
   'OUT':{'SUP':1,'ANC':0,'ESC':1,'LAND':1,'HTR':1},'ESC':{'SUP':-1,'ANC':-1,'OUT':-1,'LAND':1,'HTR':1},
   'LAND':{'SUP':-1,'ANC':-1,'OUT':-1,'ESC':-1,'HTR':0},'HTR':{'SUP':-1,'ANC':-1,'OUT':-1,'ESC':-1,'LAND':0}}
Pred=np.zeros((6,6))
for i in range(6):
    for j in range(6):
        if i!=j: Pred[i,j]=M[ROLES[i]].get(ROLES[j],0)

fig,axs=plt.subplots(1,2,figsize=(11,5.2))
# (a) predicted
norm1=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
axs[0].imshow(np.ma.masked_where(np.eye(6,dtype=bool),Pred),cmap=DIV,norm=norm1,aspect='equal')
for i in range(6):
    for j in range(6):
        if i==j: axs[0].add_patch(plt.Rectangle((j-.5,i-.5),1,1,color='0.9')); continue
        s=M[ROLES[i]].get(ROLES[j],0); txt='+' if s>0 else ('−' if s<0 else '?')
        tc='white' if s!=0 else '0.25'
        axs[0].text(j,i,txt,ha='center',va='center',fontsize=13,fontweight='bold',color=tc)
axs[0].set_title('(a) Pre-specified predicted sign',fontsize=11)
# (b) observed
vmax=np.nanmax(np.abs(E)); norm2=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
im=axs[1].imshow(np.ma.masked_where(np.eye(6,dtype=bool),E),cmap=DIV,norm=norm2,aspect='equal')
for i in range(6):
    for j in range(6):
        if i==j: axs[1].add_patch(plt.Rectangle((j-.5,i-.5),1,1,color='0.9')); continue
        val=E[i,j]
        a,b=(min(i,j),max(i,j)); pr=f'{ROLES[a]}→{ROLES[b]}'
        star='*' if fdrsig.get(pr,False) and (i<j) else ''  # 무방향쌍 유의 표시(상삼각)
        axs[1].text(j,i,f'{val:+.0f}{star}',ha='center',va='center',fontsize=8.5,
                    color='white' if abs(val)>vmax*0.55 else '0.15')
axs[1].set_title('(b) Observed effectiveness $E_{ij}$  (* FDR<0.05)',fontsize=11)
for ax in axs:
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(RLAB,fontsize=7.5); ax.set_yticklabels(RLAB,fontsize=7.5)
    ax.set_xlabel('destination role $j$'); ax.set_ylabel('origin role $i$')
    ax.set_xticks(np.arange(-.5,6,1),minor=True); ax.set_yticks(np.arange(-.5,6,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.5); ax.tick_params(which='minor',length=0)
cb=fig.colorbar(im,ax=axs[1],fraction=0.046,pad=0.04); cb.set_label('$E_{ij}=100(N_{ij}/T_{ij})$  (+: i→j net outflow)',fontsize=8)
fig.suptitle('Role-to-role directional migration: predicted vs observed pipeline (2006–2025)',fontsize=12,y=1.00)
fig.text(0.5,-0.02,f"Global sign match {perm['obs_signmatch']}/13 (null {perm['null_signmatch_mean']:.1f}, permutation p={perm['p_global']:.3f}); FDR-significant channels {perm['fdr_sig']}/13.",
         ha='center',fontsize=8.5,style='italic')
plt.tight_layout(); plt.savefig(f'{FIG}/F1_pipeline_signs.png',bbox_inches='tight'); plt.close()
print('F1 saved')

# ---------- F2 age tuning ----------
Z=np.load(f'{OUT}/tensor_static.npz',allow_pickle=True); F=Z['F']; AGES=[str(a) for a in Z['ages']]
AGELAB=['≤10','10s','20s','30s','40s','50s','60s','70+']
def eff_age(i,j):
    ii,jj=ROLES.index(i),ROLES.index(j); out=[]
    for ai in range(8):
        Fa=F[:,ai,:,:].sum(axis=0); T=Fa[ii,jj]+Fa[jj,ii]
        out.append(100*(Fa[ii,jj]-Fa[jj,ii])/T if T>0 else np.nan)
    return out
series=[('SUP→ESC (young feed)','#c0392b',eff_age('SUP','ESC')),
        ('ESC→LAND (family-stage release)','#2c6fbb',eff_age('ESC','LAND')),
        ('ESC→HTR (release, high-turnover)','#6a51a3',eff_age('ESC','HTR'))]
fig,ax=plt.subplots(figsize=(8,5))
ax.axhline(0,color='0.6',lw=1,zorder=1)
x=range(8)
for lbl,col,vals in series:
    ax.plot(x,vals,'-o',color=col,lw=2,ms=6,label=lbl,zorder=3)
    ax.text(7.05,vals[-1],lbl.split(' ')[0],color=col,fontsize=8,va='center')
ax.set_xticks(x); ax.set_xticklabels(AGELAB); ax.set_xlim(-0.3,8.6)
ax.set_xlabel('age group'); ax.set_ylabel('$E_{ij}$  (+ : net outflow i→j)')
ax.set_title('Life-course signature of the role pipeline',fontsize=12,pad=22)
ax.set_ylim(-24,36)
ax.annotate('20s peak (+28)',xy=(2,28.2),xytext=(3.0,33.5),fontsize=8,color='#c0392b',
            arrowprops=dict(arrowstyle='->',color='#c0392b',lw=1))
ax.annotate('emerges at 30s+',xy=(3,19.9),xytext=(3.4,2),fontsize=8,color='#2c6fbb',
            arrowprops=dict(arrowstyle='->',color='#2c6fbb',lw=1))
ax.legend(loc='lower right',frameon=False,fontsize=8.5)
fig.text(0.5,-0.01,'Feed channel (SUP→ESC) peaks among 20-somethings; release channels (ESC→LAND/HTR) turn on at family-formation ages. Both permutation p<0.001 in their peak age.',
         ha='center',fontsize=8,style='italic')
plt.tight_layout(); plt.savefig(f'{FIG}/F2_age_tuning.png',bbox_inches='tight'); plt.close()
print('F2 saved')

# ---------- F3 reconfiguration slope ----------
rec=json.load(open(f'{OUT}/reconfiguration_result.json')); D=rec['delta']
fig,ax=plt.subplots(figsize=(7.5,6))
order=sorted(D,key=lambda d:d['E_late'])
# 라벨 수직 충돌 완화: 후기값 기준 정렬 후 최소간격 확보한 y위치 계산
ys=[d['E_late'] for d in order]; lab_y=list(ys); gap=1.15
for k in range(1,len(lab_y)):
    if lab_y[k]-lab_y[k-1]<gap: lab_y[k]=lab_y[k-1]+gap
for k,d in enumerate(order):
    sig=d['fdr_sig']; col='#111' if sig else '0.62'; lw=2.4 if sig else 1.0
    ax.plot([0,1],[d['E_early'],d['E_late']],'-',color=col,lw=lw,zorder=4 if sig else 2)
    ax.plot(0,d['E_early'],'o',color=col,ms=5,zorder=4 if sig else 2)
    ax.plot(1,d['E_late'],'o',color=col,ms=5,zorder=4 if sig else 2)
    lab=d['pair']+('  *' if sig else '')
    ax.text(1.04,lab_y[k],lab,fontsize=7.8,va='center',color=col,fontweight='bold' if sig else 'normal')
ax.axhline(0,color='0.7',lw=0.8)
ax.set_xticks([0,1]); ax.set_xticklabels(['Early\n2008–11','Late\n2022–25'])
ax.set_xlim(-0.15,1.55); ax.set_ylabel('$E_{ij}$ (predicted-direction channel)')
ax.set_title('Reconfiguration of role-flow coupling (RQ2)',fontsize=12)
ax.spines['bottom'].set_visible(False)
def ptxt(p): return 'p<0.001' if p<0.001 else f'p={p:.3f}'
fig.text(0.5,-0.02,f"Both periods significant (early intensity {rec['early']['intensity']:+.1f} {ptxt(rec['early']['perm_p'])}; late {rec['late']['intensity']:+.1f} {ptxt(rec['late']['perm_p'])}). Bold * = FDR-significant channel change ({rec['fdr_sig']}/13).",
         ha='center',fontsize=8,style='italic')
plt.tight_layout(); plt.savefig(f'{FIG}/F3_reconfiguration.png',bbox_inches='tight'); plt.close()
print('F3 saved')
print('done')
