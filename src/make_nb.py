"""7회차: Colab에서 실행 가능한 노트북 만들기"""
import sys, nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
md("""# CNN 기반 반도체 웨이퍼 맵 불량 패턴 판별
**데이터:** WM-811K (Wu, Jang, Chen, *IEEE Trans. Semiconductor Manufacturing*, 2015) — 실제 생산 라인 웨이퍼 맵 811,457장, 그중 전문가 라벨 172,950장

**순서:** ① 데이터 열기 → ② 분포 확인 → ③ 전처리(32×32) → ④ CNN 학습(증강 전) → ⑤ 증강 후 재학습 → ⑥ 64×64로 재학습 → ⑦ 세 모델 평가·비교 → ⑧ 오분류 분석 → ⑨ Grad-CAM → ⑩ 로트 단위 SPC → ⑪ 자동/검토 분류 기준 → ⑫ Scratch 개선 실험 → ⑬ 최종 모델 저장

Colab에서는 `런타임 > 런타임 유형 변경 > T4 GPU`를 켜면 더 빨라집니다(CPU로도 실행 가능)."""),
md("## ① 데이터 열기\n`LSWMD_pkl.zip`을 업로드합니다. 구버전 pandas로 저장된 파일이라 옛 모듈 이름을 새 이름으로 연결해서 엽니다."),
code("""import glob, zipfile, pickle, numpy as np, pandas as pd
import pandas.core.indexes.base as ib

try:
    from google.colab import files
    if not glob.glob('*.pkl') and not glob.glob('*.zip'):
        files.upload()
except ImportError:
    pass  # Colab이 아닌 환경

for z in glob.glob('*.zip'):
    zipfile.ZipFile(z).extractall('.')

class OldPandasUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pandas.indexes.base':
            return ib._new_Index if name == '_new_Index' else pd.Index
        if module == 'pandas.indexes.range':
            return pd.RangeIndex
        if module == '__builtin__':
            module = 'builtins'
        return super().find_class(module, name)

with open(glob.glob('*.pkl')[0], 'rb') as f:
    df = OldPandasUnpickler(f, encoding='latin1').load()

unwrap = lambda x: x[0][0] if len(x) and len(x[0]) else None
df['failureType'] = df['failureType'].map(unwrap)
df_all = df[['waferMap', 'failureType', 'lotName', 'waferIndex']]          # 전체 811,457장 (⑩ 로트 분석용)
df = df[df.failureType.notna()][['waferMap', 'failureType']].reset_index(drop=True)   # 라벨 있는 172,950장
print(df_all.shape, df.shape)"""),
md("## ② 분포 확인\n라벨의 85%가 none(패턴 없음)이라 불균형이 매우 심합니다."),
code("""import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
CLASSES = ['none','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
WAFER = ListedColormap(['white', '#9fd3a8', '#d7263d'])   # 바깥 / 정상 / 불량

print(df.failureType.value_counts())
fig, ax = plt.subplots(3, 9, figsize=(18, 6.5))
for j, t in enumerate(CLASSES):
    s = df[df.failureType == t].sample(3, random_state=1)
    for i in range(3):
        ax[i, j].imshow(s.iloc[i].waferMap, cmap=WAFER, vmin=0, vmax=2, interpolation='nearest'); ax[i, j].axis('off')
    ax[0, j].set_title(t)
plt.tight_layout(); plt.show()"""),
md("## ③ 전처리\n- 맵 크기가 제각각이라 우선 32×32로 통일(⑥에서 64×64로 다시 해 봄)(최근접 보간 — 0/1/2 값 유지)\n- none은 10,000장만 무작위 사용, 불량 패턴은 전부 사용\n- 패턴 비율이 유지되게 학습:평가 = 8:2로 분할"),
code("""from sklearn.model_selection import train_test_split
N_NONE = 10000

def resize(m, s):
    h, w = m.shape
    r = np.minimum(((np.arange(s) + 0.5) * h / s).astype(int), h - 1)
    c = np.minimum(((np.arange(s) + 0.5) * w / s).astype(int), w - 1)
    return m[np.ix_(r, c)].astype(np.uint8)

rng = np.random.RandomState(42)
none_idx = df.index[df.failureType == 'none']
keep = np.concatenate([df.index[df.failureType != 'none'], rng.choice(none_idx, N_NONE, replace=False)])
sub = df.loc[keep]
y = sub.failureType.map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
tr_i, va_i = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y[tr], random_state=0)

def prepare(s):
    # 크기 s×s로 바꾸고 학습/검증/평가로 나누기 (나누는 방식은 크기와 관계없이 동일)
    X = np.stack([resize(m, s) for m in sub.waferMap])
    Xtr_all, ytr_all = X[tr], y[tr]
    return Xtr_all[tr_i], ytr_all[tr_i], Xtr_all[va_i], ytr_all[va_i], X[te], y[te]

Xtr, ytr, Xva, yva, Xte, yte = prepare(32)   # 학습 데이터 중 10%는 학습 곡선용 검증 데이터
print('학습', len(ytr), '검증', len(yva), '평가', len(yte))"""),
md("## ④ CNN 정의와 학습 함수\n합성곱 3층(16→32→64 채널) + 완전연결 2층, 파라미터 약 15만 개(32×32 입력 기준)"),
code("""import torch, torch.nn as nn, time
from sklearn.metrics import confusion_matrix, f1_score
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS, BS = 15, 128

class WaferCNN(nn.Module):
    def __init__(self, n=9, s=32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(64 * (s // 8) ** 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n))
        self.cam_layer = self.f[7]   # Grad-CAM에 쓸 마지막 합성곱 출력
    def forward(self, x): return self.c(self.f(x))

to_t = lambda X: torch.tensor(np.ascontiguousarray(X), dtype=torch.float32).unsqueeze(1) / 2.0

def run(Xtrain, ytrain, Xval, yval, Xtest, label, arch=None):
    torch.manual_seed(0)
    model = (arch or WaferCNN)(s=Xtrain.shape[1]).to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
    T, L, Tva = to_t(Xtrain).to(dev), torch.tensor(ytrain).to(dev), to_t(Xval).to(dev)
    def predict(Tx):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(Tx[i:i + 1024]) for i in range(0, len(Tx), 1024)]).softmax(1).cpu().numpy()
    hist, t0 = [], time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(L), device=dev); tot = 0
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]; opt.zero_grad(); loss = lossf(model(T[b]), L[b]); loss.backward(); opt.step(); tot += loss.item() * len(b)
        acc = (predict(Tva).argmax(1) == yval).mean(); hist.append((tot / len(L), acc))
        print(f'[{label}] epoch {ep + 1:2d}  loss {tot / len(L):.3f}  val_acc {acc:.3f}  ({time.time() - t0:.0f}s)')
    p = predict(to_t(Xtest).to(dev))
    return model, p, hist"""),
code("""model_base, p_base, h_base = run(Xtr, ytr, Xva, yva, Xte, '32×32 증강 전')"""),
md("## ⑤ 데이터 증강 후 재학습\n개수가 2,000장보다 적은 패턴은 90/180/270도 회전과 뒤집기로 2,000장까지 늘립니다. 웨이퍼는 원형이라 회전해도 패턴의 의미가 유지됩니다."),
code("""def augment(X, y, target=2000):
    rng = np.random.RandomState(1); Xs, ys = [X], [y]
    for c in range(9):
        idx = np.where(y == c)[0]; need = target - len(idx)
        if need <= 0: continue
        new = []
        for i in rng.choice(idx, need, replace=True):
            m = np.rot90(X[i], rng.randint(1, 4))
            if rng.rand() < 0.5: m = np.fliplr(m)
            new.append(m)
        Xs.append(np.stack(new)); ys.append(np.full(need, c))
    return np.concatenate(Xs), np.concatenate(ys)

Xaug, yaug = augment(Xtr, ytr)
print('증강 후 학습 데이터', len(yaug), np.bincount(yaug))
model_aug, p_aug, h_aug = run(Xaug, yaug, Xva, yva, Xte, '32×32 증강 후')"""),
md("## ⑥ 입력 해상도 64×64로 재학습\n32×32로 줄이면 큰 웨이퍼(평가 데이터의 71%가 32보다 큼)에서 가는 긁힘 같은 정보가 사라질 수 있습니다. 같은 분할·증강·모델 구조로 입력만 64×64로 바꿔 학습합니다(CPU 약 9분, GPU면 훨씬 빠름)."),
code("""Xtr64, ytr64, Xva64, yva64, Xte64, yte64 = prepare(64)
assert (yte64 == yte).all()   # 평가 웨이퍼는 32×32 실험과 같은 웨이퍼
Xaug64, yaug64 = augment(Xtr64, ytr64)
model_64, p_64, h_64 = run(Xaug64, yaug64, Xva64, yva64, Xte64, '64×64 증강 후')"""),
md("## ⑦ 평가와 비교\n불균형 데이터라 전체 정확도만 보면 안 되고, 패턴별 정답률(재현율)과 Macro F1을 함께 봅니다."),
code("""rows = []
for name, p in [('32×32 증강 전', p_base), ('32×32 증강 후', p_aug), ('64×64 증강 후', p_64)]:
    pred = p.argmax(1); cm = confusion_matrix(yte, pred, labels=range(9))
    print(f'{name}: 정확도 {(pred == yte).mean():.3f}, Macro F1 {f1_score(yte, pred, average="macro"):.3f}')
    rows.append(pd.Series(np.diag(cm) / cm.sum(1) * 100, index=CLASSES, name=name))
print(pd.concat(rows, axis=1).round(1))

cm = confusion_matrix(yte, p_64.argmax(1), labels=range(9)); cmn = cm / cm.sum(1, keepdims=True)
fig, a = plt.subplots(figsize=(8, 7)); a.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
for i in range(9):
    for j in range(9):
        if cm[i, j]: a.text(j, i, f'{cmn[i, j] * 100:.0f}', ha='center', va='center', color='white' if cmn[i, j] > .55 else 'black', fontsize=9)
a.set_xticks(range(9)); a.set_xticklabels(CLASSES, rotation=45, ha='right'); a.set_yticks(range(9)); a.set_yticklabels(CLASSES)
a.set_xlabel('CNN prediction'); a.set_ylabel('expert label'); a.set_title('Confusion matrix (64x64, augmented, row %)'); plt.tight_layout(); plt.show()"""),
md("## ⑧ 오분류 분석\n최종 모델(64×64)이 가장 많이 틀린 경우의 실제 웨이퍼를 확인합니다."),
code("""pred = p_64.argmax(1)
pairs = [('Scratch', 'none'), ('Loc', 'none'), ('Loc', 'Edge-Loc'), ('Scratch', 'Loc'), ('Loc', 'Scratch')]
fig, ax = plt.subplots(len(pairs), 4, figsize=(9, 2.4 * len(pairs)))
for r, (t, q) in enumerate(pairs):
    idx = np.where((yte == CLASSES.index(t)) & (pred == CLASSES.index(q)))[0]
    for k in range(4):
        ax[r, k].axis('off')
        if k < len(idx): ax[r, k].imshow(Xte64[idx[k]], cmap=WAFER, vmin=0, vmax=2, interpolation='nearest')
    ax[r, 0].set_title(f'{t} -> {q} ({len(idx)})', loc='left', fontsize=10)
plt.tight_layout(); plt.show()"""),

md("## ⑨ Grad-CAM: CNN이 웨이퍼의 어디를 보고 판단했나\n마지막 합성곱 층(16×16)의 기울기로 판단 근거 영역을 계산합니다. 밝을수록 판단에 크게 쓰인 영역이고, 하늘색 선은 불량 die 경계입니다.\n`불량 집중도`는 CAM 중 불량 die 위에 있는 비율을 불량 die 면적 비율로 나눈 값으로, 1보다 크면 불량 die를 더 많이 봤다는 뜻입니다."),
code("""import torch.nn.functional as F
def gradcam(model, Xb, cls):
    acts = {}
    h = model.cam_layer.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))
    out = model(to_t(Xb).to(dev)); h.remove()
    a = acts['a']; score = out.gather(1, torch.tensor(cls, device=dev).view(-1, 1)).sum()
    g = torch.autograd.grad(score, a)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=Xb.shape[1:], mode='bilinear', align_corners=False)[:, 0]
    return (cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)).detach().cpu().numpy()

model_64.eval(); pred64 = p_64.argmax(1)
cams = np.concatenate([gradcam(model_64, Xte64[i:i + 256], pred64[i:i + 256]) for i in range(0, len(yte), 256)])
wafer, fail = Xte64 > 0, Xte64 == 2
for k, c in enumerate(CLASSES):
    idx = np.where((yte == k) & (pred64 == k))[0]
    on_fail = (cams[idx] * fail[idx]).sum((1, 2)) / ((cams[idx] * wafer[idx]).sum((1, 2)) + 1e-8)
    area = fail[idx].sum((1, 2)) / wafer[idx].sum((1, 2))
    print(f'{c:10s} 맞힌 {len(idx):5d}장  불량 집중도 {np.mean(on_fail / np.maximum(area, 1e-3)):.2f}')

fig, ax = plt.subplots(9, 4, figsize=(10, 23))
for k, c in enumerate(CLASSES):
    ok = np.where((yte == k) & (pred64 == k))[0]; bad = np.where((yte == k) & (pred64 != k))[0]
    for j, idx in enumerate([ok, bad]):
        a1, a2 = ax[k, 2 * j], ax[k, 2 * j + 1]; a1.axis('off'); a2.axis('off')
        if len(idx) == 0: continue
        i = idx[np.argsort(-p_64[idx].max(1))][len(idx) // 3 if j == 0 else 0]
        a1.imshow(Xte64[i], cmap=WAFER, vmin=0, vmax=2, interpolation='nearest')
        a2.imshow(np.ma.masked_where(~wafer[i], cams[i]), cmap='inferno', vmin=0, vmax=1)
        a2.contour(fail[i].astype(float), levels=[0.5], colors='#35d4ff', linewidths=0.7)
        a1.set_title(('correct' if j == 0 else 'wrong') + f': {c}', fontsize=9, loc='left')
        a2.set_title(f'pred {CLASSES[pred64[i]]} {p_64[i].max() * 100:.0f}%', fontsize=9, loc='left')
plt.tight_layout(); plt.show()"""),
md("## ⑩ 로트 단위 SPC 이상 감지\n전체 811,457장에 최종 모델을 적용한 뒤 로트별로 묶습니다.\n- **주 지표: 로트 die 불량률.** 측정값이라 모델과 무관합니다. 로트당 die가 수만 개라 일반 p 관리도는 한계가 너무 좁아, 로트 간 변동을 반영하는 **Laney p\' 관리도**를 씁니다.\n- **패턴 비율 p 관리도:** 전문가 라벨이 5장 이상인 로트만 씁니다.\n- **이상 로트의 패턴:** 전문가 라벨 + 확신도 98% 이상인 모델 판별만 씁니다. 라벨 없는 웨이퍼는 분포가 달라 모델이 불안정하기 때문입니다(아래에서 확인).\n\nCPU에서는 전체 추론에 약 20분 걸립니다(GPU면 몇 분)."),
code("""maps_all = df_all.waferMap.values
n = len(maps_all); n_die = np.zeros(n, np.int32); n_fail = np.zeros(n, np.int32); pred_all = np.zeros(n, np.int8); conf_all = np.zeros(n, np.float32)
model_64.eval(); t0 = time.time()
for s0 in range(0, n, 4096):
    ch = maps_all[s0:s0 + 4096]
    n_die[s0:s0 + len(ch)] = [(m > 0).sum() for m in ch]; n_fail[s0:s0 + len(ch)] = [(m == 2).sum() for m in ch]
    x = to_t(np.stack([resize(m, 64) for m in ch])).to(dev)
    with torch.no_grad():
        pr = torch.cat([model_64(x[i:i + 512]) for i in range(0, len(x), 512)]).softmax(1).cpu().numpy()
    pred_all[s0:s0 + len(ch)] = pr.argmax(1); conf_all[s0:s0 + len(ch)] = pr.max(1)
    if s0 % (4096 * 40) == 0: print(f'{s0 + len(ch):,}/{n:,}  {time.time() - t0:.0f}s')
W_ = pd.DataFrame({'lot': df_all.lotName.values, 'n_die': n_die, 'n_fail': n_fail, 'label': df_all.failureType.values,
                   'pred': np.array(CLASSES)[pred_all], 'conf': conf_all})
W_ = W_[W_.n_die > 0].copy(); W_['lot_no'] = W_.lot.str.extract(r'(\\d+)').astype(int)
lab = W_.label.notna()
print('라벨 있는 웨이퍼: 실제 패턴 비율', round((W_[lab].label != 'none').mean(), 3), '/ 확신도 98% 이상', round((W_[lab].conf >= .98).mean(), 3))
print('라벨 없는 웨이퍼: 모델 패턴 비율', round((W_[~lab].pred != 'none').mean(), 3), '/ 확신도 98% 이상', round((W_[~lab].conf >= .98).mean(), 3))
print('die 불량률 중앙값: 라벨 있음', round((W_[lab].n_fail / W_[lab].n_die).median(), 3), '/ 라벨 없음', round((W_[~lab].n_fail / W_[~lab].n_die).median(), 3))"""),
code("""W_['pattern'] = np.where(lab, W_.label, np.where(W_.conf >= .98, W_.pred, 'uncertain'))
g = W_.groupby('lot_no')
Lt = pd.DataFrame({'lot': g.lot.first(), 'n': g.size(), 'dies': g.n_die.sum(), 'fails': g.n_fail.sum(),
                   'n_lab': g.label.apply(lambda s: s.notna().sum()),
                   'k_lab': g.label.apply(lambda s: (s.notna() & (s != 'none')).sum())}).sort_index()
Lt = Lt[Lt.n >= 5]; Lt['pd'] = Lt.fails / Lt.dies
pdbar = Lt.fails.sum() / Lt.dies.sum(); sig = np.sqrt(pdbar * (1 - pdbar) / Lt.dies)
z = (Lt.pd - pdbar) / sig; sigz = np.abs(np.diff(z.values)).mean() / 1.128      # Laney: 이동범위로 로트 간 변동 추정
Lt['ucl_classic'] = pdbar + 3 * sig; Lt['ucl'] = pdbar + 3 * sig * sigz; Lt['z'] = z / sigz
Lt['out'] = Lt.pd > Lt.ucl
print(f'관리도 로트 {len(Lt):,}개, 중심선 {pdbar:.3f}, Laney sigma_z {sigz:.1f}')
print(f"일반 p 관리도 초과 {(Lt.pd > Lt.ucl_classic).sum():,}개 vs Laney p' 초과 {Lt.out.sum():,}개")
B_ = Lt[Lt.n_lab >= 5]; pbar = B_.k_lab.sum() / B_.n_lab.sum()
print(f'라벨 기반 패턴 비율 p 관리도: 로트 {len(B_):,}개 중 {(B_.k_lab / B_.n_lab > pbar + 3 * np.sqrt(pbar * (1 - pbar) / B_.n_lab)).sum():,}개 초과')

def dominant(lot_no):
    s = W_[W_.lot_no == lot_no].pattern.value_counts().drop(['none', 'uncertain'], errors='ignore')
    return (s.index[0], int(s.iloc[0])) if len(s) else ('none', 0)
out_l = Lt[Lt.out].copy(); out_l[['dominant', 'dom_n']] = [dominant(i) for i in out_l.index]
print(out_l.sort_values('z', ascending=False).groupby('dominant').head(1)[['lot', 'n', 'pd', 'ucl', 'dominant', 'dom_n']].round(3))
pat_all = W_[~W_.pattern.isin(['none', 'uncertain'])].pattern.value_counts(normalize=True)
pat_out = W_[W_.lot_no.isin(out_l.index) & ~W_.pattern.isin(['none', 'uncertain'])].pattern.value_counts(normalize=True)
print('이상 로트에서의 패턴 농축도 (1보다 크면 이상 로트에 더 많음)'); print((pat_out / pat_all).round(2).sort_values(ascending=False))

fig, a = plt.subplots(figsize=(13, 4))
a.scatter(Lt.index, Lt.pd * 100, s=2, color='#9aa3ad'); a.scatter(Lt.index[Lt.out], Lt.pd[Lt.out] * 100, s=5, color='#c8323f')
a.axhline(pdbar * 100, color='k', lw=1); a.set_xlabel('lot number'); a.set_ylabel('lot die fail rate (%)')
a.set_title("Laney p' chart (red = above UCL)"); plt.show()"""),
md("## ⑪ 자동 분류와 엔지니어 검토 기준\n확신도 구간별 실제 정답률(보정 곡선)을 보고, 확신도 기준값마다 자동 처리 비율과 자동 처리분의 정확도를 계산합니다. 자동 처리분 정확도 98% 이상을 만족하는 가장 낮은 기준을 추천합니다."),
code("""conf64 = p_64.max(1); correct = pred64 == yte
for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, .95), (.95, .99), (.99, 1.01)]:
    m = (conf64 >= lo) & (conf64 < hi)
    print(f'확신도 {lo:.2f}~{min(hi, 1):.2f}: {m.sum():5d}장, 평균 확신도 {conf64[m].mean():.3f}, 실제 정답률 {correct[m].mean():.3f}')
rows = []
for t in [.5, .7, .8, .9, .95, .98, .99]:
    auto = conf64 >= t
    rows.append({'기준': t, '자동 처리 비율': auto.mean(), '자동 처리분 정확도': correct[auto].mean(), '자동 처리 중 오류': int((auto & ~correct).sum()),
                 '검토로 걸러진 오류 비율': (~auto & ~correct).sum() / (~correct).sum(),
                 'Scratch 검토 비율': (~auto & (yte == CLASSES.index('Scratch'))).sum() / (yte == CLASSES.index('Scratch')).sum()})
thr = pd.DataFrame(rows); print(thr.round(3))
print('추천 기준:', thr[thr['자동 처리분 정확도'] >= .98]['기준'].min())"""),
md("## ⑫ Scratch 개선 실험\n가장 약한 Scratch를 높이기 위해 두 가지를 같은 평가 웨이퍼로 비교합니다. CPU에서는 합쳐서 약 40분 걸립니다(GPU 권장).\n- **A. 불량 보존 축소:** 축소할 때 블록 안에 불량 die가 하나라도 있으면 불량으로 남겨 가는 긁힘이 지워지지 않게 합니다.\n- **B. 깊은 모델:** 합성곱 블록 4개(배치 정규화 포함)와 전역 평균 풀링을 씁니다."),
code("""def resize_failmax(m, s):
    h, w = m.shape
    out = np.maximum.reduceat(m, np.floor(np.arange(s) * h / s).astype(int), axis=0) if h >= s else m[np.minimum(((np.arange(s) + .5) * h / s).astype(int), h - 1)]
    out = np.maximum.reduceat(out, np.floor(np.arange(s) * w / s).astype(int), axis=1) if w >= s else out[:, np.minimum(((np.arange(s) + .5) * w / s).astype(int), w - 1)]
    return out.astype(np.uint8)

class DeepCNN(nn.Module):
    def __init__(self, n=9, s=64):
        super().__init__()
        blk = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.f = nn.Sequential(*blk(1, 16), nn.MaxPool2d(2), *blk(16, 32), nn.MaxPool2d(2), *blk(32, 64), nn.MaxPool2d(2), *blk(64, 96))
        self.cam_layer = self.f[-1]
        self.c = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, n))
    def forward(self, x): return self.c(self.f(x))

# A: 불량 보존 축소 (분할은 그대로, 크기 변환 방식만 바꿈)
Xfm = np.stack([resize_failmax(m, 64) for m in sub.waferMap]); Xfm_tr = Xfm[tr]
Xa, ya = augment(Xfm_tr[tr_i], y[tr][tr_i])
model_A, p_A, _ = run(Xa, ya, Xfm_tr[va_i], y[tr][va_i], Xfm[te], 'A 불량 보존 축소')
# B: 깊은 모델 (64×64 최근접, 증강 동일)
model_B, p_B, _ = run(Xaug64, yaug64, Xva64, yva64, Xte64, 'B 깊은 모델', arch=DeepCNN)

res = {}
for name, p in [('64×64 기본', p_64), ('A 불량 보존 축소', p_A), ('B 깊은 모델', p_B)]:
    pr = p.argmax(1); cm_ = confusion_matrix(yte, pr, labels=range(9))
    res[name] = pd.Series(np.diag(cm_) / cm_.sum(1) * 100, index=CLASSES)
    res[name]['정확도'] = (pr == yte).mean() * 100; res[name]['Macro F1'] = f1_score(yte, pr, average='macro') * 100
print(pd.DataFrame(res).round(1))"""),
md("## ⑬ 앙상블과 최종 모델 저장\n깊은 모델은 Scratch에 강하지만 none·Donut이 약하고, 기본 모델은 그 반대입니다. 두 모델의 확률을 평균한 앙상블을 비교합니다. 앙상블은 평가 데이터 결과를 보고 고른 것이라 수치가 조금 낙관적일 수 있습니다."),
code("""p_ens = (p_64 + p_B) / 2
pr = p_ens.argmax(1); cm_ = confusion_matrix(yte, pr, labels=range(9))
res['앙상블 (기본 + 깊은)'] = pd.Series(np.diag(cm_) / cm_.sum(1) * 100, index=CLASSES)
res['앙상블 (기본 + 깊은)']['정확도'] = (pr == yte).mean() * 100; res['앙상블 (기본 + 깊은)']['Macro F1'] = f1_score(yte, pr, average='macro') * 100
print(pd.DataFrame(res).round(1))
torch.save(model_64.state_dict(), 'wafer_cnn_base64.pt'); torch.save(model_B.state_dict(), 'wafer_cnn_deep64.pt')
print('저장: wafer_cnn_base64.pt, wafer_cnn_deep64.pt (두 모델의 확률 평균이 최종 판별)')"""),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'accelerator': 'GPU', 'colab': {'provenance': []}}
nbf.write(nb, sys.argv[1])
# 검증용: 코드 셀만 이어 붙인 스크립트
open(sys.argv[2], 'w').write('\n\n'.join(c.source for c in cells if c.cell_type == 'code'))
