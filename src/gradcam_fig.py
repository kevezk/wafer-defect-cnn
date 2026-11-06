"""4회차 보조: Grad-CAM 그림 개선 + 경계 집중도 지표"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['font.family'] = 'Noto Sans CJK KR'; plt.rcParams['axes.unicode_minus'] = False
D, O, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(D); C = list(d['classes']); X, y = d['Xte'], d['yte']
cams = np.load(f'{O}/gradcam_{TAG}.npy').astype(np.float32)
P = np.load(f'{O}/gradcam_{TAG}_pred.npy'); pred = P.argmax(1)

wafer = X > 0; fail = X == 2
# 경계 die: 상하좌우 이웃 중 정상/불량이 다른 die가 있는 웨이퍼 안 die
nb = np.zeros_like(fail)
for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
    s = np.roll(fail, (dy, dx), axis=(1, 2)); w = np.roll(wafer, (dy, dx), axis=(1, 2))
    nb |= (s != fail) & w
bound = nb & wafer
def share(mask, idx):  # 웨이퍼 안 CAM 중 mask에 있는 비율 / mask 면적 비율  (>1 이면 그 영역을 더 봄)
    cw = (cams[idx] * wafer[idx]).sum((1, 2)) + 1e-8
    cs = (cams[idx] * mask[idx]).sum((1, 2)) / cw
    area = mask[idx].sum((1, 2)) / wafer[idx].sum((1, 2))
    ok = area > 0
    return float((cs[ok] / area[ok]).mean()), float(cs[ok].mean()), float(area[ok].mean())
rows = []
for k, c in enumerate(C):
    idx = np.where((y == k) & (pred == k))[0]
    lf, sf, af = share(fail, idx); lb, sb, ab = share(bound, idx)
    rows.append({'cls': c, 'n_correct': int(len(idx)), 'fail_lift': round(lf, 2), 'boundary_lift': round(lb, 2),
                 'cam_on_fail': round(sf, 3), 'fail_area': round(af, 3), 'cam_on_boundary': round(sb, 3), 'boundary_area': round(ab, 3)})
json.dump(rows, open(f'{O}/gradcam_stats_{TAG}.json', 'w'), indent=1, ensure_ascii=False)
for r in rows: print(r)

WAFER = ListedColormap(['#ffffff', '#9fd3a8', '#d7263d'])
def cam_panel(a, i):
    a.imshow(np.ma.masked_where(~wafer[i], cams[i]), cmap='inferno', vmin=0, vmax=1, interpolation='bilinear')
    a.contour(fail[i].astype(float), levels=[0.5], colors='#35d4ff', linewidths=0.7)
    a.set_facecolor('white')
fig, ax = plt.subplots(9, 4, figsize=(10.5, 25))
for k, c in enumerate(C):
    ok = np.where((y == k) & (pred == k))[0]; ok = ok[np.argsort(-P[ok, k])][len(ok) // 3] if len(ok) else None
    bad = np.where((y == k) & (pred != k))[0]; bad = bad[np.argsort(-P[bad].max(1))][0] if len(bad) else None
    for j, (i, tag) in enumerate([(ok, '맞힘'), (bad, '틀림')]):
        a1, a2 = ax[k, 2 * j], ax[k, 2 * j + 1]; a1.axis('off'); a2.axis('off')
        if i is None: continue
        a1.imshow(X[i], cmap=WAFER, vmin=0, vmax=2, interpolation='nearest'); cam_panel(a2, i)
        a1.set_title(f'{tag}: 정답 {c}', fontsize=10, loc='left')
        a2.set_title(f'판별 {C[pred[i]]} {P[i].max() * 100:.0f}%', fontsize=10, loc='left')
fig.suptitle(f'Grad-CAM: CNN이 판단에 사용한 영역 ({TAG})\n왼쪽 2칸 맞힌 예시 · 오른쪽 2칸 틀린 예시 · 밝을수록 판단에 크게 반영 · 하늘색 선 = 불량 die 경계',
             x=0.02, ha='left', fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.975)); plt.savefig(f'{O}/fig_gradcam_{TAG}.png', dpi=110); plt.close()
fig, ax = plt.subplots(1, 9, figsize=(18, 2.6))
for k, c in enumerate(C):
    idx = np.where((y == k) & (pred == k))[0]
    m = cams[idx].mean(0); mask = wafer[idx].mean(0) < 0.5
    ax[k].imshow(np.ma.masked_where(mask, m), cmap='inferno', vmin=0); ax[k].axis('off'); ax[k].set_title(c, fontsize=11)
fig.suptitle(f'패턴별 평균 Grad-CAM (맞힌 평가 웨이퍼 평균, {TAG})', x=0.01, ha='left', fontsize=12)
plt.tight_layout(); plt.savefig(f'{O}/fig_gradcam_mean_{TAG}.png', dpi=110); plt.close()
print('ok')
