"""
Paper 4, Step 2: build the directed flow tensor of 6x6 roles x 8 age groups x periods (years)
- Inputs: _shared/raw/flow-all__sgg__{year}__{age}__ALL.json (dyadic OD, raw sgg), sgg_harmonize_map, role_assignment_229, memberships
- Inherited conventions: D1 exclude self-loops (drop ori==des flows after harmonisation) - D2 2006-2025 - D8 modal = window-mean argmax - SEED 20260716
- No role re-derivation (fixed assignments from Paper 3 as input). Outputs: static-role tensor + annual tensors + period-modal/annual-modal role tables.
"""
import json, csv, os, numpy as np
from collections import defaultdict

RAW='/mnt/user-data/uploads/migration_cache/_shared/raw'
DATA='/home/claude/paper4_work/data'
OUT='/home/claude/paper4_work/outputs'
# Korean labels mirror the raw data files / companion-study columns; do not translate.
AGES=["10세이하","10대","20대","30대","40대","50대","60대","70대이상"]
# English age labels used for all written outputs.
AGE_EN={'10세이하':'under10','10대':'10s','20대':'20s','30대':'30s','40대':'40s','50대':'50s','60대':'60s','70대이상':'70plus'}
YEARS=list(range(2006,2026))
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']
RIDX={r:i for i,r in enumerate(ROLES)}
SEED=20260716

ENGROLE={'Supplier-return':'SUP','Low-mobility anchor':'ANC','Gradual outflow':'OUT',
         'Escalator':'ESC','Landing zone':'LAND','High-turnover reception':'HTR'}
# Korean labels mirror the raw data files / companion-study columns; do not translate.
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
    """Argmax of the mean membership probability over the window [y0,y1] (D8). None if no observations in the window."""
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

    # Tensor: F[year, age, i, j] = sum of directed flows by static role, after harmonisation and self-loop exclusion
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
                if o==dd: selfl+=fl; continue          # D1 exclude self-loops
                ri=RIDX[static[o]]; rj=RIDX[static[dd]]
                F[yi,ai,ri,rj]+=fl; used+=fl
            audit.append((y,a,tot,drp,selfl,used))
    # assertion: used flow = total - dropped - self
    for (y,a,tot,drp,selfl,used) in audit:
        assert used==tot-drp-selfl, f'flow conservation violated {y} {a}'
    # diagonal (same role, different regions) sum vs off-diagonal
    diag=sum(F[:,:,k,k].sum() for k in range(6))
    offdiag=F.sum()-diag
    print(f'Tensor build complete: F shape={F.shape}, total used flow={F.sum():,}')
    print(f'  role-diagonal (same role, between regions) {diag:,} ({100*diag/F.sum():.1f}%) | role-off-diagonal {offdiag:,} ({100*offdiag/F.sum():.1f}%)')

    # Save: npz + long csv (static roles)
    np.savez(f'{OUT}/tensor_static.npz', F=F, years=np.array(YEARS), ages=np.array(AGES,dtype=object), roles=np.array(ROLES,dtype=object))
    with open(f'{OUT}/role_flow_long_static.csv','w',newline='',encoding='utf-8-sig') as fo:
        w=csv.writer(fo); w.writerow(['year','age','role_i','role_j','flow'])
        for yi,y in enumerate(YEARS):
            for ai,a in enumerate(AGES):
                for i in range(6):
                    for j in range(6):
                        if F[yi,ai,i,j]>0:
                            w.writerow([y,AGE_EN[a],ROLES[i],ROLES[j],int(F[yi,ai,i,j])])

    # Period modal role table (for RQ2): early E=(2008,2011), late L=(2022,2025)
    with open(f'{OUT}/roles_period_modal.csv','w',newline='',encoding='utf-8-sig') as fo:
        w=csv.writer(fo); w.writerow(['panel_code','static','modal_2008_2011','modal_2022_2025'])
        for c in sorted(panel):
            e=modal_window(mem[c],2008,2011) if c in mem else None
            l=modal_window(mem[c],2022,2025) if c in mem else None
            w.writerow([c,static[c],e or '',l or ''])

    # Annual modal role table (for the year-matched robustness check): argmax of that year's membership probabilities, blank for 2006-2007
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
        for (y,a,tot,drp,selfl,used) in audit: w.writerow([y,AGE_EN[a],tot,drp,selfl,used])
    print('Saved: tensor_static.npz, role_flow_long_static.csv, roles_period_modal.csv, roles_annual_modal.csv, tensor_build_audit.csv')

if __name__=='__main__':
    main()
