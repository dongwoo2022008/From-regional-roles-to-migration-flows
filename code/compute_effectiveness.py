"""
논문④ Step3: demographic effectiveness E_ij = 100*(F_ij - F_ji)/(F_ij + F_ji)
- 전체(2006-2025, 전연령) 6×6 E_ij (RQ1 주검정) + 연령별 E_ij (RQ-보조)
- 사전등록 부호패턴과 대조.
"""
import numpy as np, csv, json
ROLES=['SUP','ANC','OUT','ESC','LAND','HTR']
Z=np.load('/home/claude/paper4_work/outputs/tensor_static.npz',allow_pickle=True)
F=Z['F']  # [year, age, i, j]
YEARS=list(Z['years']); AGES=[str(a) for a in Z['ages']]

# 사전등록 부호행렬
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
    print('행 i -> 열 j : E_ij (+면 i가 j로 순유출)')
    print('        '+''.join(f'{r:>8}' for r in ROLES))
    for i in range(6):
        cells=[]
        for j in range(6):
            cells.append('     .  ' if i==j else f'{E[i,j]:+7.1f} ')
        print(f'{ROLES[i]:>6}  '+''.join(cells))

# --- RQ1 주검정: 전체 합 ---
Fall=F.sum(axis=(0,1))  # 6x6
E=eff_matrix(Fall)
pretty(E,'RQ1 전체(2006-2025, 전연령) demographic effectiveness E_ij')

# 사전등록 대조: 13개 확증 무방향쌍
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
print(f'\n--- 사전등록 확증 대조 (13 무방향쌍) ---')
print(f'{"pair":14} {"pred":4} {"E_ij":>8} {"obs":4} {"match"}')
for i,j,pred,obs,osign,match in pairs:
    print(f'{i+"→"+j:14} {pred:4} {obs:+8.1f} {osign:4} {"OK" if match else "  X"}')
print(f'\n부호 일치: {nmatch}/13  (사전 성공기준 >=11/13)')
print('판정:', 'PASS' if nmatch>=11 else 'FAIL')

# 방향예측 없음 쌍 크기 보고
print('\n--- 방향예측 없음 쌍 (크기만 보고) ---')
for (a,b) in [('ANC','OUT'),('LAND','HTR')]:
    ia,ib=ROLES.index(a),ROLES.index(b)
    print(f'  {a}↔{b}: E={E[ia,ib]:+.1f}')

# --- 연령별 E_ij: A1(SUP→ESC), A2(ESC→LAND) 강도 곡선 ---
print('\n=== 연령정합 (RQ-보조) ===')
def eff_pair_by_age(i,j):
    ii,jj=ROLES.index(i),ROLES.index(j)
    out=[]
    for ai,a in enumerate(AGES):
        Fa=F[:,ai,:,:].sum(axis=0)
        fij=Fa[ii,jj]; fji=Fa[jj,ii]; T=fij+fji
        out.append(100.0*(fij-fji)/T if T>0 else np.nan)
    return out
for (i,j,lbl) in [('SUP','ESC','A1: 공급→에스컬레이터'),('ESC','LAND','A2: 에스컬레이터→착륙지'),('ESC','HTR','에스컬레이터→고회전유입')]:
    vals=eff_pair_by_age(i,j)
    s=' '.join(f'{a}:{v:+.0f}' for a,v in zip(AGES,vals))
    print(f'  {lbl}\n    {s}')

# 저장
np.savez('/home/claude/paper4_work/outputs/effectiveness_static.npz', E_all=E, F_all=Fall,
         F_by_age=F.sum(axis=0), roles=np.array(ROLES,dtype=object), ages=np.array(AGES,dtype=object))
with open('/home/claude/paper4_work/outputs/effectiveness_all.csv','w',newline='',encoding='utf-8-sig') as fo:
    w=csv.writer(fo); w.writerow(['role_i','role_j','F_ij','F_ji','net','gross','E_ij'])
    for i in range(6):
        for j in range(6):
            if i==j: continue
            fij=int(Fall[i,j]); fji=int(Fall[j,i]); T=fij+fji
            w.writerow([ROLES[i],ROLES[j],fij,fji,fij-fji,T,f'{100.0*(fij-fji)/T:.3f}' if T>0 else ''])
# 대조결과 json
json.dump({'n_match':nmatch,'n_total':13,'verdict':'PASS' if nmatch>=11 else 'FAIL',
           'pairs':[{'i':i,'j':j,'pred':pred,'E':round(float(obs),3),'match':match} for i,j,pred,obs,_,match in pairs]},
          open('/home/claude/paper4_work/outputs/prereg_check.json','w'),ensure_ascii=False,indent=2)
print('\n저장: effectiveness_static.npz, effectiveness_all.csv, prereg_check.json')
