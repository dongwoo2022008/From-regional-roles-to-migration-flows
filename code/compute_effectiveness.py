"""
Paper 4, Step 3: demographic effectiveness E_ij = 100*(F_ij - F_ji)/(F_ij + F_ji)
- Overall (2006-2025, all ages) 6x6 E_ij (RQ1 main test) + E_ij by age group (auxiliary RQ)
- Compared against the pre-registered sign pattern.
"""
import numpy as np, csv, json
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']
Z=np.load('/home/claude/paper4_work/outputs/tensor_static.npz',allow_pickle=True)
F=Z['F']  # [year, age, i, j]
YEARS=list(Z['years']); AGES=[str(a) for a in Z['ages']]

# Pre-registered sign matrix
M={
 'SUP': {'ANC':'-','OUT':'-','ESC':'+','LAND':'+','HTR':'+'},
 'ANC': {'SUP':'+','OUT':'?','ESC':'+','LAND':'+','HTR':'+'},
 'OUT': {'SUP':'+','ANC':'?','ESC':'+','LAND':'+','HTR':'+'},
 'ESC': {'SUP':'-','ANC':'-','OUT':'-','LAND':'+','HTR':'+'},
 'LAND':{'SUP':'-','ANC':'-','OUT':'-','ESC':'-','HTR':'?'},
 'HTR': {'SUP':'-','ANC':'-','OUT':'-','ESC':'-','LAND':'?'},
}

def eff_matrix(Fsum):
    E=np.full((6,6),np.nan)
    for i in range(6):
        for j in range(6):
            if i==j: continue
            fij=Fsum[i,j]; fji=Fsum[j,i]; T=fij+fji
            if T>0: E[i,j]=100.0*(fij-fji)/T
    return E

def pretty(E,title):
    print(f'\n=== {title} ===')
    print('row i -> column j : E_ij (+ means net outflow from i to j)')
    print('        '+''.join(f'{r:>8}' for r in ROLES))
    for i in range(6):
        cells=[]
        for j in range(6):
            cells.append('     .  ' if i==j else f'{E[i,j]:+7.1f} ')
        print(f'{ROLES[i]:>6}  '+''.join(cells))

# --- RQ1 main test: overall sum ---
Fall=F.sum(axis=(0,1))  # 6x6
E=eff_matrix(Fall)
pretty(E,'RQ1 overall (2006-2025, all ages) demographic effectiveness E_ij')

# Pre-registration comparison: 13 confirmatory undirected pairs
pairs=[]
for a in range(6):
    for b in range(a+1,6):
        i,j=ROLES[a],ROLES[b]
        pred=M[i].get(j,'?')
        if pred=='?': continue
        obs=E[a,b]
        obs_sign='+' if obs>0 else ('-' if obs<0 else '0')
        match = (obs_sign==pred)
        pairs.append((i,j,pred,obs,obs_sign,match))
nmatch=sum(1 for p in pairs if p[5])
print(f'\n--- Pre-registered confirmatory comparison (13 undirected pairs) ---')
print(f'{"pair":14} {"pred":4} {"E_ij":>8} {"obs":4} {"match"}')
for i,j,pred,obs,osign,match in pairs:
    print(f'{i+"→"+j:14} {pred:4} {obs:+8.1f} {osign:4} {"OK" if match else "  X"}')
print(f'\nSign matches: {nmatch}/13  (pre-specified success criterion >=11/13)')
print('Verdict:', 'PASS' if nmatch>=11 else 'FAIL')

# Report magnitudes for pairs with no directional prediction
print('\n--- Pairs with no directional prediction (magnitude only) ---')
for (a,b) in [('ANC','OUT'),('LAND','HTR')]:
    ia,ib=ROLES.index(a),ROLES.index(b)
    print(f'  {a}↔{b}: E={E[ia,ib]:+.1f}')

# --- E_ij by age group: A1 (SUP→ESC), A2 (ESC→LAND) intensity curves ---
print('\n=== Age alignment (auxiliary RQ) ===')
def eff_pair_by_age(i,j):
    ii,jj=ROLES.index(i),ROLES.index(j)
    out=[]
    for ai,a in enumerate(AGES):
        Fa=F[:,ai,:,:].sum(axis=0)
        fij=Fa[ii,jj]; fji=Fa[jj,ii]; T=fij+fji
        out.append(100.0*(fij-fji)/T if T>0 else np.nan)
    return out
for (i,j,lbl) in [('SUP','ESC','A1: Supplier→Escalator'),('ESC','LAND','A2: Escalator→Landing zone'),('ESC','HTR','Escalator→High-turnover reception')]:
    vals=eff_pair_by_age(i,j)
    s=' '.join(f'{a}:{v:+.0f}' for a,v in zip(AGES,vals))
    print(f'  {lbl}\n    {s}')

# Save
np.savez('/home/claude/paper4_work/outputs/effectiveness_static.npz', E_all=E, F_all=Fall,
         F_by_age=F.sum(axis=0), roles=np.array(ROLES,dtype=object), ages=np.array(AGES,dtype=object))
with open('/home/claude/paper4_work/outputs/effectiveness_all.csv','w',newline='',encoding='utf-8-sig') as fo:
    w=csv.writer(fo); w.writerow(['role_i','role_j','F_ij','F_ji','net','gross','E_ij'])
    for i in range(6):
        for j in range(6):
            if i==j: continue
            fij=int(Fall[i,j]); fji=int(Fall[j,i]); T=fij+fji
            w.writerow([ROLES[i],ROLES[j],fij,fji,fij-fji,T,f'{100.0*(fij-fji)/T:.3f}' if T>0 else ''])
# Comparison-result json
json.dump({'n_match':nmatch,'n_total':13,'verdict':'PASS' if nmatch>=11 else 'FAIL',
           'pairs':[{'i':i,'j':j,'pred':pred,'E':round(float(obs),3),'match':match} for i,j,pred,obs,_,match in pairs]},
          open('/home/claude/paper4_work/outputs/prereg_check.json','w'),ensure_ascii=False,indent=2)
print('\nSaved: effectiveness_static.npz, effectiveness_all.csv, prereg_check.json')
