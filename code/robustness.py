"""
논문④ Step6: 강건성
 (R1) soft 배분: 각 지역을 단일 modal이 아닌 소속확률 벡터로 6역할에 분산.
      F = P^T A P (P[N,6]=전기간 평균 소속확률). 배정 불확실성 전파.
 (R2) 연도정합 역할: 각 연도 흐름을 그 해 modal 역할로 집계(2006-07은 정적역할).
 (R3) 대안 연령구간: 청년(20-30대) vs 그 외 로 파이프라인 강도 비교.
각 케이스 사전등록 13쌍 부호일치 + 핵심채널 E 보고. 주분석(정적 11/13)과 대조.
"""
import json,csv,numpy as np
RAW='/mnt/user-data/uploads/migration_cache/_shared/raw'
DATA='data'; OUT='outputs'
AGES=["10세이하","10대","20대","30대","40대","50대","60대","70대이상"]
YEARS=list(range(2006,2026))
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']; RIDX={r:i for i,r in enumerate(ROLES)}
M={'SUP':{'ANC':'-','OUT':'-','ESC':'+','LAND':'+','HTR':'+'},'ANC':{'SUP':'+','OUT':'?','ESC':'+','LAND':'+','HTR':'+'},'OUT':{'SUP':'+','ANC':'?','ESC':'+','LAND':'+','HTR':'+'},'ESC':{'SUP':'-','ANC':'-','OUT':'-','LAND':'+','HTR':'+'},'LAND':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','HTR':'?'},'HTR':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','LAND':'?'}}
UND=[(a,b,'+' if M[ROLES[a]].get(ROLES[b])=='+' else '-') for a in range(6) for b in range(a+1,6) if M[ROLES[a]].get(ROLES[b],'?')!='?']
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
    print(f'[{tag}] 부호일치 {nm}/13 | 핵심 {key}')
    return nm,key

# 연도별 A 캐시
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
# 주분석 재현(정적)
rv=np.array([RIDX[static[c]] for c in codes])
def role_mat(A,rvec):
    F=np.zeros((6,6))
    for i in range(6):
        ri=np.where(rvec==i)[0]
        for j in range(6):
            rj=np.where(rvec==j)[0]
            F[i,j]=A[np.ix_(ri,rj)].sum()
    return F
E0=eff(role_mat(A_all,rv)); results['주분석_정적']=report(E0,'주분석 정적 재현')[0]

# (R1) soft: 전기간 평균 소속확률
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
# Sejong pre-2012 없음 → 평균으로 채워짐(2012+). 결측지역 없음 확인
missP=[codes[i] for i in range(N) if P[i].sum()==0]
Fsoft = P.T @ A_all @ P
Esoft=eff(Fsoft); results['R1_soft']=report(Esoft,'R1 soft 배분')[0]
print(f'     (soft P 결측지역: {missP})')

# (R2) 연도정합: 각 해 modal 역할, 2006-07은 정적
annual=defaultdict(dict)
for r in csv.DictReader(open(f'{OUT}/roles_annual_modal.csv',encoding='utf-8-sig')):
    if r['role']: annual[int(r['panel_code'])][int(r['year'])]=RIDX[r['role']]
Fym=np.zeros((6,6))
for y in YEARS:
    rvy=np.array([ annual[c].get(y, RIDX[static[c]]) for c in codes ])
    Fym+=role_mat(Ayears[y],rvy)
Eym=eff(Fym); results['R2_연도정합']=report(Eym,'R2 연도정합 역할')[0]

# (R3) 대안 연령: 청년(20대,30대) vs 그외
def A_ages(agelist):
    A=np.zeros((N,N),np.int64)
    for y in YEARS:
        for a in agelist:
            for x in json.load(open(f'{RAW}/flow-all__sgg__{y}__{a}__ALL.json',encoding='utf-8'))['data']:
                o=harm(x['ori']);d=harm(x['des'])
                if o is None or d is None or o==d: continue
                A[cidx[o],cidx[d]]+=x['flow']
    return A
Ey=eff(role_mat(A_ages(['20대','30대']),rv)); results['R3_청년2030']=report(Ey,'R3 청년(20-30대)')[0]
Eo=eff(role_mat(A_ages(['50대','60대','70대이상']),rv)); results['R3_장년5070']=report(Eo,'R3 장년(50-70+)')[0]

json.dump({k:int(v) for k,v in results.items()},open(f'{OUT}/robustness_result.json','w'),ensure_ascii=False,indent=2)
print('\n요약 부호일치/13:',{k:int(v) for k,v in results.items()})
print('저장: robustness_result.json')
