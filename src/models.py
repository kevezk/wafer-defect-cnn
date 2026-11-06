"""공용: 모델 구조와 크기 변환"""
import numpy as np, torch.nn as nn

CLASSES = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']


def resize_nearest(m, s):
    """최근접 보간: 0/1/2 값 유지. 크게 줄이면 가는 불량 선이 빠질 수 있음"""
    h, w = m.shape
    r = np.minimum(((np.arange(s) + 0.5) * h / s).astype(int), h - 1)
    c = np.minimum(((np.arange(s) + 0.5) * w / s).astype(int), w - 1)
    return m[np.ix_(r, c)].astype(np.uint8)


def resize_failmax(m, s):
    """불량 보존 축소: 목표 칸에 해당하는 원본 블록에 불량이 하나라도 있으면 불량(2), 아니면 정상(1), 전부 바깥이면 0.
    원본이 목표보다 작은 방향은 최근접과 같다."""
    h, w = m.shape
    rs = np.floor(np.arange(s) * h / s).astype(int)
    cs = np.floor(np.arange(s) * w / s).astype(int)
    out = np.maximum.reduceat(m, rs, axis=0) if h >= s else m[np.minimum(((np.arange(s) + 0.5) * h / s).astype(int), h - 1)]
    out = np.maximum.reduceat(out, cs, axis=1) if w >= s else out[:, np.minimum(((np.arange(s) + 0.5) * w / s).astype(int), w - 1)]
    return out.astype(np.uint8)


class WaferCNN(nn.Module):
    """기본 모델: 합성곱 3층 + 완전연결 2층"""
    def __init__(self, n=9, s=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(64 * (s // 8) ** 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n))
        self.cam_layer = self.f[7]

    def forward(self, x):
        return self.c(self.f(x))


class DeepCNN(nn.Module):
    """깊은 모델: 합성곱 블록 4개(각 2층 + 배치 정규화) + 전역 평균 풀링"""
    def __init__(self, n=9, s=64):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                    nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.f = nn.Sequential(*block(1, 16), nn.MaxPool2d(2), *block(16, 32), nn.MaxPool2d(2),
                               *block(32, 64), nn.MaxPool2d(2), *block(64, 96))
        self.cam_layer = self.f[-1]
        self.c = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, n))

    def forward(self, x):
        return self.c(self.f(x))


ARCH = {'base': WaferCNN, 'deep': DeepCNN}
