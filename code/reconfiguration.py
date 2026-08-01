"""
Paper 4, Step 5 (RQ2): reconfiguration of the role-flow coupling structure — early (2008-2011) vs late (2022-2025)
- Each period: sum of that period's flows + that period's modal roles (D8 window-mean argmax) -> 6x6 E_ij.
- Pipeline intensity = mean of the 'predicted-direction E' over the 13 pre-registered pairs (higher = stronger pipeline).
- Per-period label permutation p (pipeline existence) + per-channel delta-E region bootstrap CI + BH-FDR.
B=1000, SEED=20260716.
"""
import json,csv,numpy as np
RAW='/mnt/user-data/uploads/migration_cache/_shared/raw'
DATA='/home/claude/paper4_work/data'; OUT='/home/claude/paper4_work/outputs'
# Korean labels mirror the raw data files / companion-study columns; do not translate.
AGES=["10세이하","10대","20대","30대","40대","50대","60대","70대이상"]
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']; RIDX={r:i for i,r in enumerate(ROLES)}
SEED=20260716; B=1000
M={ 'SUP':{'ANC':'-','OUT':'-','ESC':'+','LAND':'+','HTR':'+'},
 'ANC':{'SUP':'+','OUT':'?','ESC':'+','LAND':'+','HTR':'+'},
 'OUT':{'SUP':'+','ANC':'?','ESC':'+','LAND':'+','HTR':'+'},
 'ESC':{'SUP':'-','ANC':'-','OUT':'-','LAND':'+','HTR':'+'},
 'LAND':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','HTR':'?'},
 'HTR':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','LAND':'?'} }
# Confirmatory undirected pairs + predicted sign (+1/-1)
UND=[]
for a in range(6):
    for b in range(a+1,6):
        p=M[ROLES[a]].get(ROLES[b],'?')
        if p!='?': UND.append((a,b,1 if p=='+' else -1))

hm={};DROP=set()
for r in csv.DictReader(open(f'{DATA}/sgg_harmonize_map.csv',encoding='utf-8-sig')):
    pc=r['panel_code'].strip()
    if pc=='':DROP.add(int(r['raw_code']))
    else:hm[int(r['raw_code'])]=int(pc)
def harm(c): return None if c in DROP else hm.get(c,c)
# Period modal roles
per={}
for r in csv.DictReader(open(f'{OUT}/roles_period_modal.csv',encoding='utf-8-sig')):
    per[int(r['panel_code'])]={'E':r['modal_2008_2011'],'L':r['modal_2022_2025']}
codes=sorted(per); cidx={c:i for i,c in enumerate(codes)}; N=len(codes)

def build_A(years):
    A=np.zeros((N,N),dtype=np.int64)
    for y in years:
        for a in AGES:
            for x in json.load(open(f'{RAW}/flow-all__sgg__{y}__{a}__ALL.json',encoding='utf-8'))['data']:
                o=harm(x['ori']);d=harm(x['des'])
                if o is None or d is None or o==d: continue
                if o in cidx and d in cidx: A[cidx[o],cidx[d]]+=x['flow']
    return A
A_E=build_A(range(2008,2012)); A_L=build_A(range(2022,2026))

def rvec(period):
    return np.array([RIDX[per[c][period]] if per[c][period] else -1 for c in codes])
rE=rvec('E'); rL=rvec('L')
def role_mat(A,rv):
    F=np.zeros((6,6))
    for i in range(6):
        ri=np.where(rv==i)[0]
        for j in range(6):
            rj=np.where(rv==j)[0]
            F[i,j]=A[np.ix_(ri,rj)].sum()
    return F
def eff(F):
    E=np.full((6,6),np.nan)
    for i in range(6):
        for j in range(6):
            if i==j:continue
            T=F[i,j]+F[j,i]
            if T>0:E[i,j]=100*(F[i,j]-F[j,i])/T
    return E
def intensity(E):
    return np.mean([ (E[a,b]*s) for a,b,s in UND ])  # mean of predicted-direction components
def signmatch(E):
    return sum(1 for a,b,s in UND if (1 if E[a,b]>0 else -1)==s)

E_E=eff(role_mat(A_E,rE)); E_L=eff(role_mat(A_L,rL))
print('=== Pipeline by period ===')
for lbl,E in [('Early 2008-2011',E_E),('Late 2022-2025',E_L)]:
    print(f'{lbl}: sign matches {signmatch(E)}/13 | pipeline intensity (mean predicted-direction E) {intensity(E):+.2f}')

# Per-period label permutation p (pipeline intensity)
rng=np.random.default_rng(SEED)
def perm_p(A,rv):
    obs=intensity(eff(role_mat(A,rv)))
    valid=rv[rv>=0]
    null=np.empty(B)
    for b in range(B):
        rp=rv.copy(); perm=rng.permutation(valid); rp[rv>=0]=perm
        null[b]=intensity(eff(role_mat(A,rp)))
    return obs,float(np.mean(null>=obs)),null.mean()
oE,pE,nE=perm_p(A_E,rE); oL,pL,nL=perm_p(A_L,rL)
print(f'\nEarly intensity {oE:+.2f} vs null {nE:+.2f} p={pE:.4f}')
print(f'Late intensity {oL:+.2f} vs null {nL:+.2f} p={pL:.4f}')

# Per-channel delta E = late - early, region bootstrap CI (only regions with roles in both periods)
both=np.where((rE>=0)&(rL>=0))[0]
def eff_pair_sub(A,rv,idx,a,b):
    ri=idx[rv[idx]==a]; rj=idx[rv[idx]==b]
    fij=A[np.ix_(ri,rj)].sum(); fji=A[np.ix_(rj,ri)].sum(); T=fij+fji
    return 100*(fij-fji)/T if T>0 else np.nan
print('\n=== Per-channel reconfiguration ΔE=late-early (region bootstrap 95% CI, BH-FDR) ===')
rows=[]
for a,b,s in UND:
    dobs=eff_pair_sub(A_L,rL,both,a,b)-eff_pair_sub(A_E,rE,both,a,b)
    boot=np.empty(B)
    for k in range(B):
        samp=rng.choice(both,size=len(both),replace=True)
        boot[k]=eff_pair_sub(A_L,rL,samp,a,b)-eff_pair_sub(A_E,rE,samp,a,b)
    lo,hi=np.nanpercentile(boot,[2.5,97.5])
    # Two-sided p: extent to which the bootstrap distribution includes 0 (2*min tail)
    p=2*min(np.mean(boot<=0),np.mean(boot>=0)); p=min(p,1.0)
    rows.append((ROLES[a]+'→'+ROLES[b],eff_pair_sub(A_E,rE,both,a,b),eff_pair_sub(A_L,rL,both,a,b),dobs,lo,hi,p))
# BH-FDR
order=sorted(range(len(rows)),key=lambda k:rows[k][6]); m=len(rows); thr=0
for rank,k in enumerate(order,1):
    if rows[k][6]<=0.05*rank/m: thr=rank
sigset=set(order[:thr])
print(f'{"pair":12}{"E_early":>8}{"E_late":>8}{"ΔE":>8}{"CI95":>18}{"p":>8} FDR')
res=[]
for k,(pair,ee,el,d,lo,hi,p) in enumerate(rows):
    mark='*' if k in sigset else ' '
    print(f'{pair:12}{ee:+8.1f}{el:+8.1f}{d:+8.1f}   [{lo:+6.1f},{hi:+6.1f}]{p:8.4f}  {mark}')
    res.append({'pair':pair,'E_early':round(float(ee),2),'E_late':round(float(el),2),'dE':round(float(d),2),'ci':[round(float(lo),2),round(float(hi),2)],'p':round(float(p),4),'fdr_sig':k in sigset})
print(f'\nFDR 5% significant reconfiguration channels: {thr}/{m}')
json.dump({'early':{'signmatch':signmatch(E_E),'intensity':round(oE,3),'perm_p':pE},
           'late':{'signmatch':signmatch(E_L),'intensity':round(oL,3),'perm_p':pL},
           'n_both':int(len(both)),'delta':res,'fdr_sig':thr},
          open(f'{OUT}/reconfiguration_result.json','w'),ensure_ascii=False,indent=2)
np.savez(f'{OUT}/reconfiguration.npz',E_early=E_E,E_late=E_L,F_early=role_mat(A_E,rE),F_late=role_mat(A_L,rL))
print('Saved: reconfiguration_result.json, reconfiguration.npz')
