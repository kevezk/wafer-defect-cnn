"""4회차: Grad-CAM — CNN이 웨이퍼의 어느 영역을 보고 판단했는지"""
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams['font.family'] = 'Noto Sans CJK KR'; plt.rcParams['axes.unicode_minus'] = False
D, O = sys.argv[1], sys.argv[2]
d = np.load(D); C = list(d['classes']); X, y = d['Xte'], d['yte']; S = X.shape[1]

import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ARCH
MODEL, ARCH_NAME, TAG = sys.argv[3], sys.argv[4], sys.argv[5]
TARGET = np.load(sys.argv[6]).argmax(1) if len(sys.argv) > 6 else None   # 설명할 클래스(예: 앙상블 예측)
model = ARCH[ARCH_NAME](s=S); model.load_state_dict(torch.load(MODEL, map_location='cpu')); model.eval()
acts = {}
model.cam_layer.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))   # 마지막 합성곱 ReLU 출력 (16×16)

def gradcam(Xb, cls):
    """Xb: (N,S,S) uint8, cls: (N,) 설명할 클래스 → (N,S,S) 0~1 히트맵, 확률"""
    x = torch.tensor(Xb, dtype=torch.float32).unsqueeze(1) / 2.0
    out = model(x); a = acts['a']
    score = out.gather(1, torch.tensor(cls).view(-1, 1)).sum()
    g = torch.autograd.grad(score, a)[0]
    w = g.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(S, S), mode='bilinear', align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    return cam.detach().numpy(), out.softmax(1).detach().numpy()

# 전체 평가 데이터에 대해 예측 클래스 기준 CAM 계산
cams, probs = [], []
for i in range(0, len(y), 256):
    xb = X[i:i + 256]
    if TARGET is not None:
        pb = TARGET[i:i + 256]
    else:
        with torch.no_grad():
            pb = model(torch.tensor(xb, dtype=torch.float32).unsqueeze(1) / 2.0).argmax(1).numpy()
    c, p = gradcam(xb, pb); cams.append(c); probs.append(p)
cams = np.concatenate(cams); probs = np.concatenate(probs); pred = probs.argmax(1)
np.save(f'{O}/gradcam_{TAG}.npy', cams.astype(np.float16)); np.save(f'{O}/gradcam_{TAG}_pred.npy', probs.astype(np.float32))

print('saved', TAG)
