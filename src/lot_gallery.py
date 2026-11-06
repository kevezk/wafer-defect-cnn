"""5회차-3: 이상 로트 상위 로트의 웨이퍼 맵 모음 (그림 + 앱용 데이터)"""
import sys, os, json, pickle, numpy as np, pandas as pd, matplotlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import resize_nearest
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas.core.indexes.base as ib
plt.rcParams['font.family'] = 'Noto Sans CJK KR'; plt.rcParams['axes.unicode_minus'] = False
PKL, W, O = sys.argv[1:4]

class U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == 'pandas.indexes.base': return ib._new_Index if n == '_new_Index' else pd.Index
        if m == 'pandas.indexes.range': return pd.RangeIndex
        if m == '__builtin__' and n == 'slice': return slice
        if m in {'pandas.core.frame', 'pandas.core.internals', 'numpy', 'numpy.core.multiarray'}: return super().find_class(m, n)
        raise pickle.UnpicklingError(f'blocked {m}.{n}')

spc = json.load(open(O + '/lot_spc.json'))
top_lots = sys.argv[4].split(','); TCONF = spc['tconf']
reps = [next(x for x in spc['by_pattern'] + spc['top10'] if x['lot'] == l) for l in top_lots]
w = pd.read_parquet(W)
with open(PKL, 'rb') as f:
    df = U(f, encoding='latin1').load()
keep = df.lotName.isin(top_lots)
maps = df.loc[keep, 'waferMap']; lots = df.loc[keep, 'lotName']
wsel = w[w.lot.isin(top_lots)]
assert len(wsel) == len(maps)
gallery = {}
WAFER = ListedColormap(['#ffffff', '#9fd3a8', '#d7263d'])
for lot in top_lots:
    idx = np.where(lots.values == lot)[0]
    rows = wsel.iloc[idx]
    order = np.argsort(rows.wafer.values)
    items = []
    for k in order:
        r = rows.iloc[k]; m = resize_nearest(maps.iloc[idx[k]], 64)
        pat = r.label if pd.notna(r.label) else (r.pred if r.conf >= TCONF else '판별 보류')
        items.append({'w': int(r.wafer), 'm': ''.join(map(str, m.ravel())), 'pattern': pat,
                      'src': 'label' if pd.notna(r.label) else ('model' if r.conf >= TCONF else 'hold'), 'fail': round(float(r.n_fail / max(1, r.n_die)), 3)})
    gallery[lot] = items
json.dump(gallery, open(O + '/lot_gallery.json', 'w'))
# 그림: 1위 로트 웨이퍼 전체
lot = top_lots[0]; items = gallery[lot]
n = len(items); cols = 9; rws = int(np.ceil(n / cols))
fig, ax = plt.subplots(rws, cols, figsize=(15, 1.95 * rws + 0.6))
for i, a in enumerate(ax.ravel()):
    a.axis('off')
    if i < n:
        it = items[i]; m = np.array(list(it['m']), dtype=int).reshape(64, 64)
        a.imshow(m, cmap=WAFER, vmin=0, vmax=2, interpolation='nearest')
        a.set_title(f"#{it['w']} {it['pattern']}{'' if it['src'] == 'label' else '*'}", fontsize=9,
                    color='#c8323f' if it['pattern'] not in ('none', '판별 보류') else '#555')
t = reps[0]
fig.suptitle(f"이상 로트 {lot}: die 불량률 {t['die_fail'] * 100:.0f}% (관리 상한 {t['ucl'] * 100:.0f}%), 주요 패턴 {t['dominant']} · *는 확신도 98% 이상 모델 판별, 나머지는 전문가 라벨",
             x=0.01, ha='left', fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(O + '/fig_toplot.png', dpi=110); plt.close()
print('ok', top_lots)
