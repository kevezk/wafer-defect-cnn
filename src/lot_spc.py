"""5회차-2: 로트 단위 SPC
 (A) die 불량률 Laney p' 관리도 — 측정값만 사용(모델과 무관), 주 감시 지표
 (B) 패턴 불량 웨이퍼 비율 p 관리도 — 전문가 라벨이 5장 이상인 로트만
 (C) 이상 로트의 패턴 = 전문가 라벨 + 확신도 98% 이상인 모델 판별만 사용 (라벨 없는 웨이퍼는 분포가 달라 모델이 불안정)
"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK KR'; plt.rcParams['axes.unicode_minus'] = False
W, O = sys.argv[1], sys.argv[2]
TCONF = 0.98
CAUSE = {'Center': '증착·식각 균일도, CMP 중앙부 압력', 'Donut': '열처리·챔버 온도 분포, 증착 두께 편차',
         'Edge-Loc': '웨이퍼 핸들링·클램프 접촉, 가장자리 국부 오염', 'Edge-Ring': '스핀 코팅 가장자리, edge bead 제거, 식각 가장자리 마진',
         'Loc': '파티클 덩어리, 국부 결함, 레티클 결함', 'Random': '파티클 오염, 장비·클린룸 오염',
         'Scratch': '이송 로봇·핸들링 긁힘, CMP 긁힘', 'Near-full': '공정 단계 누락, 장비 고장 등 치명적 이상', 'none': '-', '판별 보류': '-'}

w = pd.read_parquet(W)
w = w[w.n_die > 0].copy()
w['lot_no'] = w.lot.str.extract(r'(\d+)').astype(int)
lab = w.label.notna(); un = ~lab
# 모델 신뢰성 점검 (라벨 있는 웨이퍼 vs 없는 웨이퍼)
shift = {'labeled_true_patterned': float((w[lab].label != 'none').mean()), 'labeled_pred_patterned': float((w[lab].pred != 'none').mean()),
         'unlabeled_pred_patterned': float((w[un].pred != 'none').mean()), 'unlabeled_pred_scratch': float((w[un].pred == 'Scratch').mean()),
         'labeled_conf98': float((w[lab].conf >= TCONF).mean()), 'unlabeled_conf98': float((w[un].conf >= TCONF).mean()),
         'labeled_fail_median': float((w[lab].n_fail / w[lab].n_die).median()), 'unlabeled_fail_median': float((w[un].n_fail / w[un].n_die).median())}
# 웨이퍼 패턴: 라벨 > 확신도 높은 모델 판별 > 판별 보류
w['pattern'] = np.where(lab, w.label, np.where(w.conf >= TCONF, w.pred, '판별 보류'))

g = w.groupby('lot_no')
L = pd.DataFrame({'lot': g.lot.first(), 'n': g.size(), 'dies': g.n_die.sum(), 'fails': g.n_fail.sum(),
                  'n_lab': g.label.apply(lambda s: s.notna().sum()),
                  'k_lab': g.label.apply(lambda s: ((s.notna()) & (s != 'none')).sum())}).sort_index()
L = L[L.n >= 5]
# (A) Laney p'
L['pd'] = L.fails / L.dies
pdbar = L.fails.sum() / L.dies.sum()
sig = np.sqrt(pdbar * (1 - pdbar) / L.dies)
z = (L.pd - pdbar) / sig
sigz = np.abs(np.diff(z.values)).mean() / 1.128
L['ucl_classic'] = pdbar + 3 * sig
L['ucl'] = pdbar + 3 * sig * sigz
L['z'] = z / sigz
L['out_classic'] = L.pd > L.ucl_classic
L['out'] = L.pd > L.ucl
# (B) 라벨 기반 p 관리도
B = L[L.n_lab >= 5].copy()
pbar = B.k_lab.sum() / B.n_lab.sum()
B['p'] = B.k_lab / B.n_lab
B['ucl_p'] = pbar + 3 * np.sqrt(pbar * (1 - pbar) / B.n_lab)
B['out_p'] = B.p > B.ucl_p
L = L.join(B[['p', 'ucl_p', 'out_p']])

# (C) 이상 로트의 패턴 구성
def comp(lot_no):
    s = w[w.lot_no == lot_no].pattern.value_counts()
    pat = s.drop(['none', '판별 보류'], errors='ignore')
    return (pat.index[0] if len(pat) else ('판별 보류' if s.get('판별 보류', 0) > s.get('none', 0) else 'none'),
            int(pat.iloc[0]) if len(pat) else 0, int(s.get('판별 보류', 0)))
out = L[L.out].copy()
out[['dominant', 'dom_n', 'uncertain']] = [comp(i) for i in out.index]
out = out.sort_values('z', ascending=False)
# 연속 이상 로트 구간 (로트 번호가 이어지는 이상 로트 묶음)
nums = out.index.sort_values().values
runs, start = [], nums[0]
for a_, b_ in zip(nums[:-1], nums[1:]):
    if b_ != a_ + 1: runs.append((start, a_)); start = b_
runs.append((start, nums[-1]))
runs = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:5]
outs = out.sort_index()
runs_info = [{'from': int(r[0]), 'to': int(r[1]), 'lots': int(r[1] - r[0] + 1),
              'dominant': outs.loc[r[0]:r[1]].dominant.value_counts().index[0],
              'die_fail': round(float(outs.loc[r[0]:r[1]].fails.sum() / outs.loc[r[0]:r[1]].dies.sum()), 3)} for r in runs]
# 패턴별 대표 이상 로트 (패턴마다 가장 심한 로트 1개)
rep = out[~out.dominant.isin(['none', '판별 보류'])].groupby('dominant').head(1).sort_values('z', ascending=False)
pat_out = w[w.lot_no.isin(out.index) & ~w.pattern.isin(['none', '판별 보류'])].pattern.value_counts(normalize=True)
pat_all = w[~w.pattern.isin(['none', '판별 보류'])].pattern.value_counts(normalize=True)
enrich = pd.DataFrame({'all': pat_all, 'abnormal': pat_out}).fillna(0); enrich['ratio'] = enrich.abnormal / enrich['all']

def row(r):
    return {'lot': r.lot, 'lot_no': int(r.name), 'wafers': int(r.n), 'die_fail': round(float(r.pd), 3), 'ucl': round(float(r.ucl), 3),
            'z': round(float(r.z), 1), 'dominant': r.dominant, 'dominant_n': int(r.dom_n), 'uncertain': int(r.uncertain),
            'labeled': int(r.n_lab), 'cause': CAUSE.get(r.dominant, '-')}
summary = {'wafers': int(len(w)), 'wafers_labeled': int(lab.sum()), 'lots_total': int(w.lot_no.nunique()), 'lots_charted': int(len(L)),
           'pdbar': float(pdbar), 'laney_sigma_z': float(sigz), 'out_classic': int(L.out_classic.sum()), 'out_laney': int(L.out.sum()),
           'p_lots': int(len(B)), 'pbar_label': float(pbar), 'out_p_label': int(B.out_p.sum()),
           'out_both': int((L.out & L.out_p.fillna(False).astype(bool)).sum()),
           'shift': shift, 'tconf': TCONF, 'runs': runs_info, 'dominant_counts': out.dominant.value_counts().to_dict(),
           'enrichment': enrich.round(3).to_dict(orient='index'),
           'top10': [row(r) for _, r in out.head(10).iterrows()], 'by_pattern': [row(r) for _, r in rep.iterrows()]}
json.dump(summary, open(O + '/lot_spc.json', 'w'), indent=1, ensure_ascii=False)
L.to_csv(O + '/lot_table.csv'); out.to_csv(O + '/lot_abnormal.csv')
print(json.dumps({k: v for k, v in summary.items() if k not in ('top10', 'enrichment', 'by_pattern')}, ensure_ascii=False, indent=1))
for t in summary['by_pattern']: print(t)
print(enrich.round(3))

# 그림
INK, MUTED, RED, GRID = '#222', '#9aa3ad', '#c8323f', '#e6e6e6'
fig, ax = plt.subplots(2, 1, figsize=(13, 8.4))
a = ax[0]
a.scatter(L.index, L.pd * 100, s=2, color=MUTED, lw=0, label='로트')
a.scatter(L.index[L.out], L.pd[L.out] * 100, s=5, color=RED, lw=0, label="Laney p' 상한 초과")
a.axhline(pdbar * 100, color=INK, lw=1)
a.text(L.index.max(), pdbar * 100 + 2, f'중심선 {pdbar * 100:.1f}%', ha='right', fontsize=9)
for r in runs_info[:1]:
    a.annotate(f"연속 이상 로트 {r['lots']}개\n(lot{r['from']}~{r['to']}, {r['dominant']})", xy=((r['from'] + r['to']) / 2, r['die_fail'] * 100),
               xytext=(20, -40), textcoords='offset points', fontsize=9, arrowprops=dict(arrowstyle='-', color=INK, lw=0.7))
a.set_ylabel('로트 die 불량률 (%)'); a.set_xlabel('로트 번호')
a.set_title(f"die 불량률 Laney p' 관리도: 로트 {len(L):,}개 중 {int(L.out.sum()):,}개 상한 초과 (일반 p 관리도였다면 {int(L.out_classic.sum()):,}개)", loc='left')
a.legend(frameon=False, loc='upper left', markerscale=4)
roll = L.out.astype(int).rolling(400).sum(); end = int(roll.idxmax()); Z = L.loc[:end].tail(400)
a = ax[1]
a.plot(Z.index, Z.pd * 100, color=MUTED, lw=0.8, marker='o', ms=2.5)
a.plot(Z.index, Z.ucl * 100, color=INK, lw=1, ls='--', label="관리 상한 (Laney p')")
a.axhline(pdbar * 100, color=INK, lw=1, label='중심선')
a.scatter(Z.index[Z.out], Z.pd[Z.out] * 100, s=22, color=RED, zorder=3, label='이상 로트')
a.set_ylabel('로트 die 불량률 (%)'); a.set_xlabel('로트 번호 (이상 로트가 가장 많이 몰린 400개 구간)')
a.legend(frameon=False, loc='upper left', fontsize=9)
for x in ax: x.spines[['top', 'right']].set_visible(False); x.grid(axis='y', color=GRID); x.set_axisbelow(True)
plt.tight_layout(); plt.savefig(O + '/fig_pchart.png', dpi=120); plt.close()

# 모델 신뢰성 점검 그림: 확신도 분포 (라벨 있음 vs 없음)
fig, a = plt.subplots(figsize=(8, 3.6))
bins = np.linspace(0.2, 1, 33)
a.hist(w[lab].conf, bins=bins, density=True, histtype='step', lw=2, color=MUTED, label='라벨 있는 웨이퍼')
a.hist(w[un].conf, bins=bins, density=True, histtype='step', lw=2, color='#2a78d6', label='라벨 없는 웨이퍼')
a.axvline(TCONF, color=INK, ls='--', lw=1); a.text(TCONF - 0.01, a.get_ylim()[1] * 0.9, '자동 분류 기준 98%', ha='right', fontsize=9)
a.set_xlabel('모델 확신도'); a.set_ylabel('밀도')
a.set_title(f"확신도 98% 이상: 라벨 있음 {shift['labeled_conf98'] * 100:.0f}% vs 라벨 없음 {shift['unlabeled_conf98'] * 100:.0f}%", loc='left')
a.legend(frameon=False, loc='upper left'); a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(O + '/fig_shift.png', dpi=120); plt.close()
print('ok')
