"""
논문④ Step4-보조: 연령슬라이스 permutation — 생애과정 특이성 확증
A1: 20대에서 SUP→ESC(공급→에스컬레이터) 순유출이 유의한가?
A2: 30대+에서 ESC→LAND(에스컬레이터→착륙지) 순유출이 유의한가?
역할라벨 permutation(크기보존), B=1000, SEED=20260716.
"""
import json,csv,numpy as np
RAW='/mnt/user-data/uploads/migration_cache/_shared/raw'
DATA='/home/claude/paper4_work/data'; OUT='/home/claude/paper4_work/outputs'
AGES=["10세이하","10대","20대","30대","40대","50대","60대","70대이상"]
YEARS=range(2006,2026)
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']; RIDX={r:i for i,r in enumerate(ROLES)}
SEED=20260716; B=1000
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
rvec=np.array([RIDX[role[c]] for c in codes])

def build_A(agelist):
    A=np.zeros((N,N),dtype=np.int64)
    for y in YEARS:
        for a in agelist:
            for x in json.load(open(f'{RAW}/flow-all__sgg__{y}__{a}__ALL.json',encoding='utf-8'))['data']:
                o=harm(x['ori']);d=harm(x['des'])
                if o is None or d is None or o==d: continue
                A[cidx[o],cidx[d]]+=x['flow']
    return A
def eff_pair(A,rv,i,j):
    ri=np.where(rv==i)[0]; rj=np.where(rv==j)[0]
    fij=A[np.ix_(ri,rj)].sum(); fji=A[np.ix_(rj,ri)].sum(); T=fij+fji
    return 100*(fij-fji)/T if T>0 else np.nan

tests=[('20대','SUP','ESC','A1 공급→에스컬레이터 @20대 (예측 강한 +)'),
       ('30대','ESC','LAND','A2 에스컬레이터→착륙지 @30대 (예측 강한 +)'),
       ('20대','ESC','LAND','대조: ESC→LAND @20대 (예측 약함)'),
       ('40대','ESC','LAND','ESC→LAND @40대 (예측 +)')]
rng=np.random.default_rng(SEED)
out=[]
cacheA={}
for age,i,j,lbl in tests:
    if age not in cacheA: cacheA[age]=build_A([age])
    A=cacheA[age]; ii,jj=RIDX[i],RIDX[j]
    eobs=eff_pair(A,rvec,ii,jj)
    null=np.array([eff_pair(A,rng.permutation(rvec),ii,jj) for _ in range(B)])
    # 예측 방향이 + 이므로 단측 p = P(null >= eobs); 부호 자체는 이미 관측
    p_one=float(np.mean(null>=eobs))
    p_two=float(np.mean(np.abs(null)>=abs(eobs)))
    print(f'{lbl}\n   E_obs={eobs:+.1f} | 귀무평균 {np.nanmean(null):+.2f} | p(단측)={p_one:.4f} p(양측)={p_two:.4f}')
    out.append({'age':age,'pair':f'{i}→{j}','E':round(float(eobs),2),'p_one':round(p_one,4),'p_two':round(p_two,4)})
json.dump(out,open(f'{OUT}/permutation_age.json','w'),ensure_ascii=False,indent=2)
print('\n저장: permutation_age.json')
