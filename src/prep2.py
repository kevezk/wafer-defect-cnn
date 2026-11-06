"""전처리 (크기·축소 방식 선택): prep.py와 같은 표본·분할"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split
from models import CLASSES, resize_nearest, resize_failmax
SRC, OUT, S, MODE = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rs = resize_failmax if MODE == 'failmax' else resize_nearest
df = pd.read_pickle(SRC)[['waferMap', 'failureType', 'lotName']]
rng = np.random.RandomState(42)
none_idx = df.index[df.failureType == 'none']
keep = np.concatenate([df.index[df.failureType != 'none'], rng.choice(none_idx, 10000, replace=False)])
sub = df.loc[keep]
X = np.stack([rs(m, S) for m in sub.waferMap])
y = sub.failureType.map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
np.savez_compressed(OUT, Xtr=X[tr], ytr=y[tr], Xte=X[te], yte=y[te], classes=np.array(CLASSES), idx_te=np.asarray(keep)[te])
print(OUT, X.shape)
