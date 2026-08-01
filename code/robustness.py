"""
Paper 4, Step 6: robustness
 (R1) Soft allocation: spread each region over the 6 roles by its membership-probability vector rather than a single modal role.
      F = P^T A P (P[N,6] = full-period mean membership probabilities). Propagates assignment uncertainty.
 (R2) Year-matched roles: aggregate each year's flows using that year's modal roles (static roles for 2006-07).
 (R3) Alternative age bands: compare pipeline intensity for the young (20s-30s) vs the rest.
Each case reports the 13 pre-registered pair sign matches + key-channel E. Compared with the main analysis (static, 11/13).
"""
import json,csv,numpy as np
RAW='/mnt/user-data/uploads/migration_cache/_shared/raw'
DATA='data'; OUT='outputs'
# Korean labels mirror the raw data files / companion-study columns; do not translate.
AGES=["10세이하","10대","20대","30대","40대","50대","60대","70대이상"]
YEARS=list(range(2006,2026))
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']; RIDX={r:i for i,r in enumerate(ROLES)}
M={'SUP':{'ANC':'-','OUT':'-','ESC':'+','LAND':'+','HTR':'+'},'ANC':{'SUP':'+','OUT':'?','ESC':'+','LAND':'+','HTR':'+'},'OUT':{'SUP':'+','ANC':'?','ESC':'+','LAND':'+','HTR':'+'},'ESC':{'SUP':'-','ANC':'-','OUT':'-','LAND':'+','HTR':'+'},'LAND':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','HTR':'?'},'HTR':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','LAND':'?'}}
UND=[(a,b,'+' if M[ROLES[a]].get(ROLES[b])=='+' else '-') for a in range(6) for b in range(a+1,6) if M[ROLES[a]].get(ROLES[b],'?')!='?']
# Korean labels mirror the raw data files / companion-study columns; do not translate.
MEMCOL={'p_공급·회귀형':'SUP','p_안정저이동형':'ANC','p_완만유출형':'OUT','p_에스컬레이터형':'ESC','p_착륙지형':'LAND','p_고회전유입형':'HTR'}
ENG={'Supplier-return':'SUP','Low-mobility anchor':'ANC','Gradual outflow':'OUT','Escalator':'ESC','Landing zone':'LAND','High-turnover reception':'HTR'}

hm={};DROP=set()
for r in csv.DictReader(open(f'{DATA}/sgg_harmonize_map.csv',encoding='utf-8-sig')):
    pc=r['panel_code'].strip()
    if pc=='':DROP.add(int(r['raw_code']))
    else:hm[int(r['raw_code'])]=int(pc)
def harm(c): return None if c in DROP else hm.get(c,c)
static={int(r['panel_code']):ENG[r['role']] for r in csv.DictReader(open(f'{DATA}/role_assignment_229.csv',encoding='utf-8-sig'))}
codes=sorted(static); cidx={c:i for i,c in enumerate(codes)}; N=len(codes)

def eff(F):
    E=np.full((6,6),np.nan)
    for i in range(6):
        for j in range(6):
            if i==j:continue
            T=F[i,j]+F[j,i]
            if T>0:E[i,j]=100*(F[i,j]-F[j,i])/T
    return E
def report(E,tag):
    nm=sum(1 for a,b,s in UND if ('+' if E[a,b]>0 else '-')==s)
    key={f'{ROLES[a]}→{ROLES[b]}':round(float(E[a,b]),1) for a,b,_ in
         [(RIDX['SUP'],RIDX['ESC'],0),(RIDX['ESC'],RIDX['LAND'],0),(RIDX['ESC'],RIDX['HTR'],0),(RIDX['OUT'],RIDX['ESC'],0)]}
    print(f'[{tag}] sign matches {nm}/13 | key channels {key}')
    return nm,key

# Cache of annual A matrices
def build_A_year(y):
    A=np.zeros((N,N),np.int64)
    for a in AGES:
        for x in json.load(open(f'{RAW}/flow-all__sgg__{y}__{a}__ALL.json',encoding='utf-8'))['data']:
            o=harm(x['ori']);d=harm(x['des'])
            if o is None or d is None or o==d: continue
            A[cidx[o],cidx[d]]+=x['flow']
    return A
Ayears={y:build_A_year(y) for y in YEARS}
A_all=sum(Ayears.values())

results={}
# Reproduce the main analysis (static)
rv=np.array([RIDX[static[c]] for c in codes])
def role_mat(A,rvec):
    F=np.zeros((6,6))
    for i in range(6):
        ri=np.where(rvec==i)[0]
        for j in range(6):
            rj=np.where(rvec==j)[0]
            F[i,j]=A[np.ix_(ri,rj)].sum()
    return F
E0=eff(role_mat(A_all,rv)); results['main_static']=report(E0,'Main analysis, static (reproduction)')[0]

# (R1) soft: full-period mean membership probabilities
mem=csv.DictReader(open(f'{DATA}/memberships.csv',encoding='utf-8-sig'))
from collections import defaultdict
acc=defaultdict(lambda:np.zeros(6)); cnt=defaultdict(int)
pcols=None
for r in csv.DictReader(open(f'{DATA}/memberships.csv',encoding='utf-8-sig')):
    if pcols is None: pcols=[k for k in r if k.startswith('p_')]
    c=int(r['panel_code'])
    v=np.array([float(r[k]) for k in pcols]); order=[MEMCOL[k] for k in pcols]
    # reorder to ROLES
    vr=np.array([v[order.index(rr)] for rr in ROLES])
    acc[c]+=vr; cnt[c]+=1
P=np.zeros((N,6))
for c in codes:
    P[cidx[c]] = acc[c]/cnt[c] if cnt[c] else 0
# Sejong has no pre-2012 data -> filled with the mean (2012+). Confirm no regions are missing
missP=[codes[i] for i in range(N) if P[i].sum()==0]
Fsoft = P.T @ A_all @ P
Esoft=eff(Fsoft); results['R1_soft_allocation']=report(Esoft,'R1 soft allocation')[0]
print(f'     (regions missing soft P: {missP})')

# (R2) year-matched: each year's modal roles, static for 2006-07
annual=defaultdict(dict)
for r in csv.DictReader(open(f'{OUT}/roles_annual_modal.csv',encoding='utf-8-sig')):
    if r['role']: annual[int(r['panel_code'])][int(r['year'])]=RIDX[r['role']]
Fym=np.zeros((6,6))
for y in YEARS:
    rvy=np.array([ annual[c].get(y, RIDX[static[c]]) for c in codes ])
    Fym+=role_mat(Ayears[y],rvy)
Eym=eff(Fym); results['R2_year_matched']=report(Eym,'R2 year-matched roles')[0]

# (R3) alternative age bands: young (20s, 30s) vs the rest
def A_ages(agelist):
    A=np.zeros((N,N),np.int64)
    for y in YEARS:
        for a in agelist:
            for x in json.load(open(f'{RAW}/flow-all__sgg__{y}__{a}__ALL.json',encoding='utf-8'))['data']:
                o=harm(x['ori']);d=harm(x['des'])
                if o is None or d is None or o==d: continue
                A[cidx[o],cidx[d]]+=x['flow']
    return A
# Age-list values are Korean raw-file labels (do not translate).
Ey=eff(role_mat(A_ages(['20대','30대']),rv)); results['R3_young_20s_30s']=report(Ey,'R3 young (20s-30s)')[0]
Eo=eff(role_mat(A_ages(['50대','60대','70대이상']),rv)); results['R3_older_50s_70plus']=report(Eo,'R3 older (50s-70+)')[0]

json.dump({k:int(v) for k,v in results.items()},open(f'{OUT}/robustness_result.json','w'),ensure_ascii=False,indent=2)
print('\nSummary of sign matches /13:',{k:int(v) for k,v in results.items()})
print('Saved: robustness_result.json')
