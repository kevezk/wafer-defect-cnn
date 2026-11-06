"""5회차-1: 811,457장 전체에 최종 모델 적용 → 웨이퍼별 표 저장 (메모리 절약을 위해 나눠서 처리)"""
import sys, os, gc, time, pickle, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CLASSES, ARCH, resize_nearest, resize_failmax
import pandas.core.indexes.base as ib

PKL, MODEL, ARCH_NAME, RESIZE, OUT = sys.argv[1:6]
rs = resize_failmax if RESIZE == 'failmax' else resize_nearest

class OldPandasUnpickler(pickle.Unpickler):
    OK = {'pandas.core.frame', 'pandas.core.internals', 'numpy', 'numpy.core.multiarray'}
    def find_class(self, module, name):
        if module == 'pandas.indexes.base':
            return ib._new_Index if name == '_new_Index' else pd.Index
        if module == 'pandas.indexes.range':
            return pd.RangeIndex
        if module == '__builtin__' and name == 'slice':
            return slice
        if module in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')

t0 = time.time()
with open(PKL, 'rb') as f:
    df = OldPandasUnpickler(f, encoding='latin1').load()
unwrap = lambda x: x[0][0] if len(x) and len(x[0]) else None
label = df['failureType'].map(unwrap).values
lot = df['lotName'].values; widx = df['waferIndex'].values.astype(int)
maps = df['waferMap'].values
del df; gc.collect()
print('loaded', len(maps), f'{time.time() - t0:.0f}s', flush=True)

model = ARCH[ARCH_NAME](s=64); model.load_state_dict(torch.load(MODEL, map_location='cpu')); model.eval()
n = len(maps)
n_die = np.zeros(n, np.int32); n_fail = np.zeros(n, np.int32)
pred = np.zeros(n, np.int8); conf = np.zeros(n, np.float32)
B = 4096
for s0 in range(0, n, B):
    chunk = maps[s0:s0 + B]
    n_die[s0:s0 + len(chunk)] = [int((m > 0).sum()) for m in chunk]
    n_fail[s0:s0 + len(chunk)] = [int((m == 2).sum()) for m in chunk]
    x = torch.tensor(np.stack([rs(m, 64) for m in chunk]), dtype=torch.float32).unsqueeze(1) / 2.0
    with torch.no_grad():   # 512장씩 계산 (한 번에 4096장이면 첫 합성곱 출력만 1GB라 메모리 부족)
        p = torch.cat([model(x[i:i + 512]) for i in range(0, len(x), 512)]).softmax(1).numpy()
    pred[s0:s0 + len(chunk)] = p.argmax(1); conf[s0:s0 + len(chunk)] = p.max(1)
    if (s0 // B) % 20 == 0:
        print(f'{s0 + len(chunk):,}/{n:,}  {time.time() - t0:.0f}s', flush=True)

pd.DataFrame({'lot': lot, 'wafer': widx, 'n_die': n_die, 'n_fail': n_fail, 'label': label,
              'pred': [CLASSES[i] for i in pred], 'conf': conf}).to_parquet(OUT)
print('saved', OUT, f'{time.time() - t0:.0f}s')
