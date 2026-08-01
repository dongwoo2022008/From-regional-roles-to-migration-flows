"""
논문④ Step2: 6×6 역할 × 8연령 × 기간(연도) 방향성 흐름 텐서 빌드
- 입력: _shared/raw/flow-all__sgg__{year}__{age}__ALL.json (dyadic OD, raw sgg), sgg_harmonize_map, role_assignment_229, memberships
- 규약 상속: D1 자기루프 제외(조화 후 ori==des flow 제외) · D2 2006-2025 · D8 modal=창평균 argmax · SEED 20260716
- 역할 재도출 없음(③ 고정 배정 입력). 산출: 정적역할 텐서 + 연도별 + 기간modal/연도modal 역할표.
"""
import json, csv, os, numpy as np
from collections import defaultdict

RAW='/mnt/user-data/uploads/migration_cache/_shared/raw'
DATA='/home/claude/paper4_work/data'
OUT='/home/claude/paper4_work/outputs'
AGES=["10세이하","10대","20대","30대","40대","50대","60대","70대이상"]
YEARS=list(range(2006,2026))
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']
RIDX={r:i for i,r in enumerate(ROLES)}
SEED=20260716

ENGROLE={'Supplier-return':'SUP','Low-mobility anchor':'ANC','Gradual outflow':'OUT',
         'Escalator':'ESC','Landing zone':'LAND','High-turnover reception':'HTR'}
MEMCOL={'p_공급·회귀형':'SUP','p_안정저이동형':'ANC','p_완만유출형':'OUT',
        'p_에스컬레이터형':'ESC','p_착륙지형':'LAND','p_고회전유입형':'HTR'}

def load_harmonize():
    hm={}; DROP=set()
    for r in csv.DictReader(open(f'{DATA}/sgg_harmonize_map.csv',encoding='utf-8-sig')):
        raw=int(r['raw_code']); pc=r['panel_code'].strip()
        if pc=='': DROP.add(raw)
        else: hm[raw]=int(pc)
    return hm,DROP

def load_static():
    d={}
    for r in csv.DictReader(open(f'{DATA}/role_assignment_229.csv',encoding='utf-8-sig')):
        d[int(r['panel_code'])]=ENGROLE[r['role']]
    return d

def load_memberships():
    mem=defaultdict(dict)
    rows=list(csv.DictReader(open(f'{DATA}/memberships.csv',encoding='utf-8-sig')))
    pcols=[c for c in rows[0].keys() if c.startswith('p_')]
    for r in rows:
        c=int(r['panel_code']); y=int(r['year'])
        mem[c][y]={MEMCOL[k]:float(r[k]) for k in pcols}
    return mem

def modal_window(mem_c, y0, y1):
    """창구간 [y0,y1] 평균 소속확률 argmax (D8). 해당 창에 관측 없으면 None."""
    vs=[mem_c[y] for y in mem_c if y0<=y<=y1]
    if not vs: return None
    agg=defaultdict(float)
    for v in vs:
        for k,x in v.items(): agg[k]+=x
    return max(agg,key=agg.get)

def main():
    hm,DROP=load_harmonize()
    static=load_static()
    mem=load_memberships()
    panel=set(static)
    def harm(c):
        if c in DROP: return None
        return hm.get(c,c)

    # 텐서: F[year, age, i, j] = 조화·자기루프제외 후 정적역할 기준 방향성 흐름 합
    F=np.zeros((len(YEARS),len(AGES),6,6),dtype=np.int64)
    audit=[]
    for yi,y in enumerate(YEARS):
        for ai,a in enumerate(AGES):
            f=f'{RAW}/flow-all__sgg__{y}__{a}__ALL.json'
            d=json.load(open(f,encoding='utf-8'))['data']
            tot=0; used=0; selfl=0; drp=0
            for x in d:
                fl=x['flow']; tot+=fl
                o=harm(x['ori']); dd=harm(x['des'])
                if o is None or dd is None: drp+=fl; continue
                if o==dd: selfl+=fl; continue          # D1 자기루프 제외
                ri=RIDX[static[o]]; rj=RIDX[static[dd]]
                F[yi,ai,ri,rj]+=fl; used+=fl
            audit.append((y,a,tot,drp,selfl,used))
    # assertion: 사용 흐름 = 총 - drop - self
    for (y,a,tot,drp,selfl,used) in audit:
        assert used==tot-drp-selfl, f'흐름 보존 위반 {y} {a}'
    # 대각(동일역할, 서로 다른 지역) 합 vs 비대각
    diag=sum(F[:,:,k,k].sum() for k in range(6))
    offdiag=F.sum()-diag
    print(f'텐서 빌드 완료 F shape={F.shape}, 총 사용흐름={F.sum():,}')
    print(f'  역할-대각(동일역할 지역간) {diag:,} ({100*diag/F.sum():.1f}%) | 역할-비대각 {offdiag:,} ({100*offdiag/F.sum():.1f}%)')

    # 저장: npz + long csv (정적역할)
    np.savez(f'{OUT}/tensor_static.npz', F=F, years=np.array(YEARS), ages=np.array(AGES,dtype=object), roles=np.array(ROLES,dtype=object))
    with open(f'{OUT}/role_flow_long_static.csv','w',newline='',encoding='utf-8-sig') as fo:
        w=csv.writer(fo); w.writerow(['year','age','role_i','role_j','flow'])
        for yi,y in enumerate(YEARS):
            for ai,a in enumerate(AGES):
                for i in range(6):
                    for j in range(6):
                        if F[yi,ai,i,j]>0:
                            w.writerow([y,a,ROLES[i],ROLES[j],int(F[yi,ai,i,j])])

    # 기간 modal 역할표 (RQ2용): 초기 E=(2008,2011), 후기 L=(2022,2025)
    with open(f'{OUT}/roles_period_modal.csv','w',newline='',encoding='utf-8-sig') as fo:
        w=csv.writer(fo); w.writerow(['panel_code','static','modal_2008_2011','modal_2022_2025'])
        for c in sorted(panel):
            e=modal_window(mem[c],2008,2011) if c in mem else None
            l=modal_window(mem[c],2022,2025) if c in mem else None
            w.writerow([c,static[c],e or '',l or ''])

    # 연도별 modal 역할표 (연도-정합 강건성용): argmax(그 해 소속확률), 2006-2007은 공란
    with open(f'{OUT}/roles_annual_modal.csv','w',newline='',encoding='utf-8-sig') as fo:
        w=csv.writer(fo); w.writerow(['panel_code','year','role'])
        for c in sorted(panel):
            for y in YEARS:
                if c in mem and y in mem[c]:
                    v=mem[c][y]; w.writerow([c,y,max(v,key=v.get)])
                else:
                    w.writerow([c,y,''])

    # audit csv
    with open(f'{OUT}/tensor_build_audit.csv','w',newline='',encoding='utf-8-sig') as fo:
        w=csv.writer(fo); w.writerow(['year','age','total_flow','dropped','self_loop','used_interrole'])
        for row in audit: w.writerow(row)
    print('저장: tensor_static.npz, role_flow_long_static.csv, roles_period_modal.csv, roles_annual_modal.csv, tensor_build_audit.csv')

if __name__=='__main__':
    main()
