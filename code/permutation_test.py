"""
Paper 4, Step 4: role-label permutation test + BH-FDR
Null hypothesis: the roles measured in Paper 3 do not organise directional flows.
  -> Randomly reassigning region-role labels (preserving each role's region count) would yield the same directional pipeline.
Observed statistics:
  (1) Number of sign matches over the 13 pre-registered pairs (observed 11) -> test for a global pipeline
  (2) Cell-wise E_ij : proportion of |E| >= |E_obs| (two-sided) -> BH-FDR 5% (cells with predicted direction)
B=1000, SEED=20260716. (Not community detection or embedding: node-attribute label permutation.)
"""
import json,csv,numpy as np
from collections import defaultdict
RAW='/mnt/user-data/uploads/migration_cache/_shared/raw'
DATA='/home/claude/paper4_work/data'; OUT='/home/claude/paper4_work/outputs'
# Korean labels mirror the raw data files / companion-study columns; do not translate.
AGES=["10세이하","10대","20대","30대","40대","50대","60대","70대이상"]
YEARS=range(2006,2026)
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']; RIDX={r:i for i,r in enumerate(ROLES)}
SEED=20260716; B=1000
M={ 'SUP':{'ANC':'-','OUT':'-','ESC':'+','LAND':'+','HTR':'+'},
 'ANC':{'SUP':'+','OUT':'?','ESC':'+','LAND':'+','HTR':'+'},
 'OUT':{'SUP':'+','ANC':'?','ESC':'+','LAND':'+','HTR':'+'},
 'ESC':{'SUP':'-','ANC':'-','OUT':'-','LAND':'+','HTR':'+'},
 'LAND':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','HTR':'?'},
 'HTR':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','LAND':'?'} }

# harmonize + roles
hm={};DROP=set()
for r in csv.DictReader(open(f'{DATA}/sgg_harmonize_map.csv',encoding='utf-8-sig')):
    pc=r['panel_code'].strip()
    if pc=='':DROP.add(int(r['raw_code']))
    else:hm[int(r['raw_code'])]=int(pc)
ENG={'Supplier-return':'SUP','Low-mobility anchor':'ANC','Gradual outflow':'OUT','Escalator':'ESC','Landing zone':'LAND','High-turnover reception':'HTR'}
role={}
for r in csv.DictReader(open(f'{DATA}/role_assignment_229.csv',encoding='utf-8-sig')):
    role[int(r['panel_code'])]=ENG[r['role']]
def harm(c): return None if c in DROP else hm.get(c,c)
codes=sorted(role); cidx={c:i for i,c in enumerate(codes)}; N=len(codes)

# region-level directed flow matrix A[N,N] (pooled 2006-2025, all ages, self excluded)
A=np.zeros((N,N),dtype=np.int64)
for y in YEARS:
    for a in AGES:
        for x in json.load(open(f'{RAW}/flow-all__sgg__{y}__{a}__ALL.json',encoding='utf-8'))['data']:
            o=harm(x['ori']);d=harm(x['des'])
            if o is None or d is None or o==d: continue
            A[cidx[o],cidx[d]]+=x['flow']
print(f'Region directed matrix A: {N}x{N}, total {A.sum():,}')

rvec=np.array([RIDX[role[c]] for c in codes])
def role_matrix(rv):
    F=np.zeros((6,6))
    for i in range(6):
        rows=np.where(rv==i)[0]
        for j in range(6):
            cols=np.where(rv==j)[0]
            F[i,j]=A[np.ix_(rows,cols)].sum()
    return F
def eff(F):
    E=np.full((6,6),np.nan)
    for i in range(6):
        for j in range(6):
            if i==j: continue
            T=F[i,j]+F[j,i]
            if T>0: E[i,j]=100*(F[i,j]-F[j,i])/T
    return E

# Confirmatory directed cells (those with a predicted sign): per direction (i,j,predsign)
cells=[]
for a in range(6):
    for b in range(6):
        if a==b: continue
        p=M[ROLES[a]].get(ROLES[b],'?')
        if p!='?': cells.append((a,b,p))

E_obs=eff(role_matrix(rvec))
def signmatch(E):
    n=0
    for a in range(6):
        for b in range(a+1,6):
            p=M[ROLES[a]].get(ROLES[b],'?')
            if p=='?': continue
            os_='+' if E[a,b]>0 else '-'
            if os_==p: n+=1
    return n
obs_match=signmatch(E_obs)

rng=np.random.default_rng(SEED)
null_match=np.zeros(B,int)
null_E=np.zeros((B,6,6))
for b in range(B):
    rp=rng.permutation(rvec)
    Eb=eff(role_matrix(rp))
    null_match[b]=signmatch(Eb)
    null_E[b]=np.nan_to_num(Eb,nan=0.0)

p_match=float(np.mean(null_match>=obs_match))
print(f'\n[Global pipeline test]')
print(f'Observed sign matches {obs_match}/13 | null mean {null_match.mean():.2f} (95% {np.percentile(null_match,2.5):.0f}-{np.percentile(null_match,97.5):.0f}) | p={p_match:.4f}')

# Cell-wise two-sided p + BH-FDR (over the 20 directed cells with predicted direction; antisymmetric, so equivalent to 13 undirected pairs)
# Test at the undirected-pair level (i<j)
und=[]
for a in range(6):
    for b in range(a+1,6):
        p=M[ROLES[a]].get(ROLES[b],'?')
        if p=='?': continue
        eobs=E_obs[a,b]
        pv=float(np.mean(np.abs(null_E[:,a,b])>=abs(eobs)))
        und.append((pv,a,b,eobs,p))
und.sort()
m=len(und)
thr=0
for rank,(pv,a,b,eobs,p) in enumerate(und,1):
    if pv<=0.05*rank/m: thr=rank
sig=set(id(x) for x in und[:thr])
print(f'\n[Cell-wise directional significance] BH-FDR 5% (m={m} undirected pairs)')
print(f'{"pair":12}{"E_obs":>8}{"pred":>6}{"null|E|mean":>12}{"p":>8}  FDR')
res=[]
for k,(pv,a,b,eobs,p) in enumerate(und):
    mark='*' if k<thr else ' '
    nm=np.mean(np.abs(null_E[:,a,b]))
    print(f'{ROLES[a]+"→"+ROLES[b]:12}{eobs:+8.1f}{p:>6}{nm:12.2f}{pv:8.4f}   {mark}')
    res.append({'pair':f'{ROLES[a]}→{ROLES[b]}','E':round(float(eobs),2),'pred':p,'p':round(pv,4),'fdr_sig':k<thr})
print(f'\nFDR 5% significant undirected pairs: {thr}/{m}')

json.dump({'obs_signmatch':int(obs_match),'null_signmatch_mean':float(null_match.mean()),
           'p_global':p_match,'B':B,'SEED':SEED,'fdr_sig':thr,'m':m,'cells':res},
          open(f'{OUT}/permutation_result.json','w'),ensure_ascii=False,indent=2)
np.savez(f'{OUT}/permutation_null.npz',null_match=null_match,null_E=null_E,E_obs=E_obs,A_rolesum=role_matrix(rvec))
print('\nSaved: permutation_result.json, permutation_null.npz')
