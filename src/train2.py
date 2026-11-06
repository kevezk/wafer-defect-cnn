"""2~3회차: CNN 학습/평가. mode=base(증강 없음) | aug(소수 클래스 회전·뒤집기 증강)"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ARCH
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

DATA, MODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ARCH_NAME = sys.argv[4] if len(sys.argv) > 4 else 'base'
EPOCHS, BS, MIN_PER_CLASS = 15, 128, 2000
torch.manual_seed(0); np.random.seed(0)

d = np.load(DATA)
classes = list(d['classes'])
Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
# 학습 데이터 중 10%는 학습 곡선용 검증 데이터 (평가 데이터는 마지막에만 사용)
tr, va = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=0)
Xva, yva = Xtr[va], ytr[va]
Xtr, ytr = Xtr[tr], ytr[tr]

def augment(X, y, target):
    """클래스별 개수가 target보다 적으면 90/180/270도 회전 + 좌우·상하 뒤집기로 채움"""
    rng = np.random.RandomState(1)
    Xs, ys = [X], [y]
    for c in range(len(classes)):
        idx = np.where(y == c)[0]
        need = target - len(idx)
        if need <= 0:
            continue
        pick = rng.choice(idx, need, replace=True)
        new = []
        for i in pick:
            m = np.rot90(X[i], rng.randint(1, 4))
            if rng.rand() < 0.5:
                m = np.fliplr(m)
            new.append(m)
        Xs.append(np.stack(new)); ys.append(np.full(need, c))
    return np.concatenate(Xs), np.concatenate(ys)

if MODE == 'aug':
    Xtr, ytr = augment(Xtr, ytr, MIN_PER_CLASS)

def to_t(X):  # 0=바깥, 1=정상, 2=불량 → 0, 0.5, 1
    return torch.tensor(np.ascontiguousarray(X), dtype=torch.float32).unsqueeze(1) / 2.0

model = ARCH[ARCH_NAME](s=Xtr.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
Ttr, Ltr = to_t(Xtr), torch.tensor(ytr)
Tva, Tte = to_t(Xva), to_t(Xte)

def predict(T):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(T[i:i + 1024]) for i in range(0, len(T), 1024)]).softmax(1).numpy()

hist = []
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Ltr))
    tot = 0
    for i in range(0, len(perm), BS):
        b = perm[i:i + BS]
        opt.zero_grad()
        loss = lossf(model(Ttr[b]), Ltr[b]); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    pva = predict(Tva)
    hist.append({'epoch': ep + 1, 'train_loss': tot / len(Ltr),
                 'val_loss': float(nn.functional.nll_loss(torch.tensor(np.log(pva + 1e-9)), torch.tensor(yva)).item()),
                 'val_acc': float((pva.argmax(1) == yva).mean())})
    print(hist[-1], f'{time.time() - t0:.0f}s', flush=True)

p = predict(Tte); pred = p.argmax(1)
cm = confusion_matrix(yte, pred, labels=range(9))
res = {'mode': MODE, 'n_train': int(len(ytr)), 'acc': float((pred == yte).mean()),
       'macro_f1': float(f1_score(yte, pred, average='macro')),
       'per_class_recall': {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes)},
       'confusion': cm.tolist(), 'history': hist, 'train_sec': time.time() - t0,
       'train_counts': np.bincount(ytr, minlength=9).tolist()}
json.dump(res, open(OUT + '.json', 'w'), indent=1)
np.save(OUT + '_proba.npy', p)
torch.save(model.state_dict(), OUT + '.pt')
print('ACC', res['acc'], 'macroF1', res['macro_f1'])
