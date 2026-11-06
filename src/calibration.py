"""6회차: 보정 곡선 + 신뢰도 기준값별 자동/검토 분류"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK KR'; plt.rcParams['axes.unicode_minus'] = False
D, PROBA, O, TAG = sys.argv[1:5]
d = np.load(D); C = list(d['classes']); y = d['yte']
P = np.load(PROBA); pred = P.argmax(1); conf = P.max(1); correct = pred == y

# 보정 곡선 (확신도 구간별 실제 정답률)
bins = np.array([0, .5, .6, .7, .8, .9, .95, .99, 1.0001])
cal = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf >= lo) & (conf < hi)
    if m.sum():
        cal.append({'range': f'{lo:.2f}~{min(hi, 1):.2f}', 'n': int(m.sum()), 'mean_conf': float(conf[m].mean()), 'acc': float(correct[m].mean())})
ece = sum(c['n'] * abs(c['mean_conf'] - c['acc']) for c in cal) / len(y)

# 기준값별 자동 처리
rows = []
for t in [0, .5, .7, .8, .9, .95, .98, .99]:
    auto = conf >= t
    per = {c: float(correct[auto & (y == k)].sum() / max(1, (y == k).sum())) for k, c in enumerate(C)}  # 패턴별: 자동으로 맞힌 비율
    sent = {c: float((~auto & (y == k)).sum() / max(1, (y == k).sum())) for k, c in enumerate(C)}      # 패턴별: 검토로 넘어간 비율
    rows.append({'t': t, 'auto_rate': float(auto.mean()), 'auto_acc': float(correct[auto].mean()),
                 'review_n': int((~auto).sum()), 'review_err_share': float((~correct & ~auto).sum() / max(1, (~correct).sum())),
                 'auto_errors': int((~correct & auto).sum()), 'sent_to_review': sent})
# 추천: 자동 처리분 정확도 98% 이상을 만족하는 가장 낮은 기준값
ok = [r for r in rows if r['auto_acc'] >= 0.98]
rec = min(ok, key=lambda r: r['t']) if ok else rows[-1]
out = {'tag': TAG, 'ece': ece, 'calibration': cal, 'thresholds': rows, 'recommended': rec['t'], 'target_acc': 0.98}
json.dump(out, open(f'{O}/calib_{TAG}.json', 'w'), indent=1, ensure_ascii=False)
print('ECE', round(ece, 4))
for c in cal: print(c)
for r in rows: print(r['t'], round(r['auto_rate'], 3), round(r['auto_acc'], 4), r['review_n'], r['auto_errors'], round(r['review_err_share'], 3))
print('recommended', rec['t'], {k: round(v, 2) for k, v in rec['sent_to_review'].items()})

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
a = ax[0]
a.plot([0.5, 1], [0.5, 1], color='#9aa3ad', ls='--', lw=1, label='완벽한 보정')
a.plot([c['mean_conf'] for c in cal], [c['acc'] for c in cal], marker='o', color='#2a78d6', lw=2, label='모델')
for c in cal: a.annotate(f"{c['n']:,}장", (c['mean_conf'], c['acc']), textcoords='offset points', xytext=(6, -12), fontsize=8, color='#555')
a.set_xlabel('모델 확신도 (구간 평균)'); a.set_ylabel('실제 정답률'); a.set_xlim(0.45, 1.01); a.set_ylim(0.3, 1.02)
a.set_title(f'보정 곡선 (ECE {ece * 100:.1f}%p)', loc='left'); a.legend(frameon=False, loc='upper left')
a = ax[1]
ts = [r['t'] for r in rows if r['t'] > 0]
a.plot([r['auto_rate'] * 100 for r in rows if r['t'] > 0], [r['auto_acc'] * 100 for r in rows if r['t'] > 0], marker='o', color='#2a78d6', lw=2)
for r in rows:
    if r['t'] > 0:
        a.annotate(f"≥{r['t']:.2f}", (r['auto_rate'] * 100, r['auto_acc'] * 100), textcoords='offset points', xytext=(6, 4), fontsize=8,
                   color='#c8323f' if r['t'] == rec['t'] else '#555', weight='bold' if r['t'] == rec['t'] else 'normal')
a.axhline(98, color='#9aa3ad', ls='--', lw=1); a.text(0.99, 98.08, '목표 정확도 98%', fontsize=8, color='#555', ha='right', transform=a.get_yaxis_transform())
a.set_xlabel('자동 처리 비율 (%)'); a.set_ylabel('자동 처리분 정확도 (%)')
a.set_title(f'기준값별 자동 처리: 추천 기준 ≥{rec["t"]:.2f}', loc='left')
for x in ax: x.spines[['top', 'right']].set_visible(False); x.grid(color='#e6e6e6'); x.set_axisbelow(True)
plt.tight_layout(); plt.savefig(f'{O}/fig_calibration_{TAG}.png', dpi=120); plt.close()
