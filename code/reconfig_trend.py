"""
미결1: 이동창·3기간 재편 (RQ2 검정력 보강)
(A) 연도별 파이프라인 강도 2008-2025 (연도modal 역할, 그 해 흐름) + OLS 선형추세 + 지역부트스트랩 CI.
(B) 3기간(2008-13/2014-19/2020-25, 6년창) 기간modal 재편: 강도·부호일치 + 채널별 단조추세.
파이프라인 강도 = 사전등록 13쌍의 예측방향 E 평균.  SEED=20260716, B=1000.
"""
import json,csv,numpy as np
from collections import defaultdict
RAW='/mnt/user-data/uploads/migration_cache/_shared/raw'
DATA='data'; OUT='outputs'
AGES=["10세이하","10대","20대","30대","40대","50대","60대","70대이상"]
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']; RIDX={r:i for i,r in enumerate(ROLES)}
SEED=20260716; B=1000
M={'SUP':{'ANC':'-','OUT':'-','ESC':'+','LAND':'+','HTR':'+'},'ANC':{'SUP':'+','OUT':'?','ESC':'+','LAND':'+','HTR':'+'},'OUT':{'SUP':'+','ANC':'?','ESC':'+','LAND':'+','HTR':'+'},'ESC':{'SUP':'-','ANC':'-','OUT':'-','LAND':'+','HTR':'+'},'LAND':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','HTR':'?'},'HTR':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','LAND':'?'}}
UND=[(a,b,1 if M[ROLES[a]].get(ROLES[b])=='+' else -1) for a in range(6) for b in range(a+1,6) if M[ROLES[a]].get(ROLES[b],'?')!='?']
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
# memberships
mem=defaultdict(dict)
rows=list(csv.DictReader(open(f'{DATA}/memberships.csv',encoding='utf-8-sig')))
pcols=[c for c in rows[0].keys() if c.startswith('p_')]
for r in rows:
    c=int(r['panel_code']); y=int(r['year'])
    mem[c][y]={MEMCOL[k]:float(r[k]) for k in pcols}
def modal_win(c,y0,y1):
    vs=[mem[c][y] for y in mem.get(c,{}) if y0<=y<=y1]
    if not vs: return None
    agg=defaultdict(float)
    for v in vs:
        for k,x in v.items(): agg[k]+=x
    return max(agg,key=agg.get)

def build_A(years):
    A=np.zeros((N,N),np.int64)
    for y in years:
        for a in AGES:
            for x in json.load(open(f'{RAW}/flow-all__sgg__{y}__{a}__ALL.json',encoding='utf-8'))['data']:
                o=harm(x['ori']);d=harm(x['des'])
                if o is None or d is None or o==d: continue
                A[cidx[o],cidx[d]]+=x['flow']
    return A
def pair_e(A,rv,idx,a,b):
    ri=idx[rv[idx]==a]; rj=idx[rv[idx]==b]
    f=A[np.ix_(ri,rj)].sum(); g=A[np.ix_(rj,ri)].sum(); T=f+g
    return 100*(f-g)/T if T>0 else np.nan
def intensity(A,rv,idx):
    return np.nanmean([pair_e(A,rv,idx,a,b)*s for a,b,s in UND])

# ---------- (A) 연도별 강도 + 추세 ----------
Ay={y:build_A([y]) for y in range(2008,2026)}
annual={}  # (c,y)->role idx (그 해 argmax)
for c in codes:
    for y in range(2008,2026):
        if c in mem and y in mem[c]:
            v=mem[c][y]; annual[(c,y)]=RIDX[max(v,key=v.get)]
allidx=np.arange(N)
years=list(range(2008,2026))
def rvec_year(y):
    return np.array([annual.get((c,y), RIDX[static[c]]) for c in codes])
inten_series=np.array([intensity(Ay[y],rvec_year(y),allidx) for y in years])
# OLS slope
xs=np.array(years)-2008
slope,icpt=np.polyfit(xs,inten_series,1)
# region bootstrap: 지역 재표집 → 전 연도 강도·기울기 재계산
rng=np.random.default_rng(SEED)
boot_slopes=np.empty(B)
for bi in range(B):
    samp=rng.choice(allidx,N,replace=True)
    ser=np.array([intensity(Ay[y],rvec_year(y),samp) for y in years])
    boot_slopes[bi]=np.polyfit(xs,ser,1)[0]
lo,hi=np.percentile(boot_slopes,[2.5,97.5]); pslope=2*min(np.mean(boot_slopes<=0),np.mean(boot_slopes>=0))
print('=== (A) 연도별 파이프라인 강도 2008-2025 ===')
print(' '.join(f'{y}:{v:+.1f}' for y,v in zip(years,inten_series)))
print(f'OLS 추세 기울기 {slope:+.3f}/년 (부트스트랩 95% CI [{lo:+.3f},{hi:+.3f}], p={min(pslope,1.0):.4f})')
print(f'2008 예측강도 {icpt:+.2f} → 2025 예측강도 {icpt+slope*17:+.2f}')

# ---------- (B) 3기간 6년창 ----------
PERIODS=[('2008-2013',range(2008,2014),(2008,2013)),
         ('2014-2019',range(2014,2020),(2014,2019)),
         ('2020-2025',range(2020,2026),(2020,2025))]
print('\n=== (B) 3기간(6년창) 재편 ===')
per_int={}; per_E={}
Ap={}
for nm,yrs,(w0,w1) in PERIODS:
    A=build_A(yrs); Ap[nm]=A
    rv=np.array([ (RIDX[modal_win(c,w0,w1)] if modal_win(c,w0,w1) else -1) for c in codes])
    idx=np.where(rv>=0)[0]
    per_int[nm]=intensity(A,rv,idx)
    E=np.full((6,6),np.nan)
    for a in range(6):
        for b in range(6):
            if a!=b: E[a,b]=pair_e(A,rv,idx,a,b)
    per_E[nm]=E
    sm=sum(1 for a,b,s in UND if (1 if E[a,b]>0 else -1)==s)
    print(f'  {nm}: 강도 {per_int[nm]:+.2f} | 부호일치 {sm}/13 | 지역 {len(idx)}')
# 채널별 3기간 값 + 단조추세
print('\n  채널별 3기간 E (단조↑/↓ 표시):')
names=[p[0] for p in PERIODS]
chan=[]
for a,b,s in UND:
    vals=[per_E[nm][a,b] for nm in names]
    mono='↑' if vals[0]<vals[1]<vals[2] else ('↓' if vals[0]>vals[1]>vals[2] else '·')
    chan.append((f'{ROLES[a]}→{ROLES[b]}',vals,mono))
for pr,vals,mono in chan:
    print(f'   {pr:12} {vals[0]:+6.1f} {vals[1]:+6.1f} {vals[2]:+6.1f}  {mono}')
nmono=sum(1 for _,_,m in chan if m!='·')
print(f'  단조추세 채널: {nmono}/13')

json.dump({'annual':{'years':years,'intensity':[round(float(v),3) for v in inten_series],
                     'slope':round(float(slope),4),'ci':[round(float(lo),4),round(float(hi),4)],'p':round(float(min(pslope,1.0)),4)},
           'three_period':{nm:round(float(per_int[nm]),3) for nm in names},
           'channels':[{'pair':pr,'vals':[round(float(v),2) for v in vals],'mono':m} for pr,vals,m in chan]},
          open(f'{OUT}/reconfig_trend.json','w'),ensure_ascii=False,indent=2)
np.savez(f'{OUT}/reconfig_trend.npz',years=np.array(years),inten=inten_series,boot_slopes=boot_slopes,
         E_p1=per_E[names[0]],E_p2=per_E[names[1]],E_p3=per_E[names[2]])
print('\n저장: reconfig_trend.json, reconfig_trend.npz')
