"""앱 빌드 (6회차 이후): 모델 구조 일반화 + Grad-CAM + 자동/검토 기준 + 로트 SPC"""
import sys, os, json, base64, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ARCH

ap = argparse.ArgumentParser()
ap.add_argument('--data'); ap.add_argument('--nets', nargs='+', help='arch=model.pt (여러 개면 확률 평균)'); ap.add_argument('--tag')
ap.add_argument('--proba'); ap.add_argument('--out'); ap.add_argument('--template'); ap.add_argument('--desc')
ap.add_argument('--models', nargs='+', help='표시이름=결과json')
ap.add_argument('--note')
a = ap.parse_args()
O = os.path.dirname(a.proba)
d = np.load(a.data); C = list(d['classes']); X, y = d['Xte'], d['yte']; S = X.shape[1]
P = np.load(a.proba)
def export(model):
    mods = [m for m in list(model.f) + list(model.c)]
    layers, chunks, i = [], [], 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach().clone(); b = m.bias.detach().clone() if m.bias is not None else torch.zeros(m.out_channels)
            if i + 1 < len(mods) and isinstance(mods[i + 1], nn.BatchNorm2d):
                bn = mods[i + 1]; sc = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
                w = w * sc.view(-1, 1, 1, 1); b = (b - bn.running_mean) * sc + bn.bias.detach(); i += 1
            relu = i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU)
            layers.append({'type': 'conv', 'cin': m.in_channels, 'cout': m.out_channels, 'relu': bool(relu)})
            chunks += [w.numpy().ravel(), b.numpy().ravel()]
        elif isinstance(m, nn.MaxPool2d): layers.append({'type': 'pool'})
        elif isinstance(m, nn.AdaptiveAvgPool2d): layers.append({'type': 'gap'})
        elif isinstance(m, nn.Linear):
            relu = i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU)
            layers.append({'type': 'dense', 'nin': m.in_features, 'nout': m.out_features, 'relu': bool(relu)})
            chunks += [m.weight.detach().numpy().ravel(), m.bias.detach().numpy().ravel()]
        i += 1
    flat = np.concatenate(chunks).astype('<f4')
    return {'layers': layers, 'weights': base64.b64encode(flat.tobytes()).decode()}, flat.size

nets, n_params = [], 0
for spec in a.nets:
    arch, path = spec.split('=', 1)
    m = ARCH[arch](s=S); m.load_state_dict(torch.load(path, map_location='cpu')); m.eval()
    e, k = export(m); nets.append(e); n_params += k
flat_size = n_params

# ---- 평가 웨이퍼 샘플 (패턴별 최대 25장) + Grad-CAM ----
cams = np.load(f'{O}/gradcam_{a.tag}.npy').astype(np.float32)
rng = np.random.RandomState(7)
pick = np.concatenate([rng.choice(np.where(y == c)[0], min(25, (y == c).sum()), replace=False) for c in range(9)])
rng.shuffle(pick)
samples = [{'m': ''.join(map(str, X[i].ravel())), 'y': int(y[i]), 'p': [round(float(v), 5) for v in P[i]],
            'cam': base64.b64encode(np.round(np.clip(cams[i], 0, 1) * 255).astype(np.uint8).tobytes()).decode()} for i in pick]

# ---- 성능 비교 ----
models = []
for spec in a.models:
    name, path = spec.split('=', 1); r = json.load(open(path))
    models.append({'name': name, 'acc': r['acc'], 'f1': r['macro_f1'], 'recall': r['per_class_recall']})
F = json.load(open(a.models[-1].split('=', 1)[1]))

# ---- 자동/검토 기준 ----
cb = json.load(open(f'{O}/calib_{a.tag}.json'))
rec = next(r for r in cb['thresholds'] if r['t'] == cb['recommended'])
over = [c for c in cb['calibration'] if c['mean_conf'] - c['acc'] > 0.03]
calib = {'recommended': cb['recommended'], 'thresholds': cb['thresholds'],
         'note': (f"확률이 {cb['recommended'] * 100:.0f}% 이상인 웨이퍼만 자동으로 분류하면 평가 웨이퍼의 {rec['auto_rate'] * 100:.0f}%가 자동 처리되고 "
                  f"그 정확도는 {rec['auto_acc'] * 100:.1f}%입니다. 모델이 틀린 웨이퍼의 {rec['review_err_share'] * 100:.0f}%는 검토 대상으로 걸러집니다. "
                  f"모델은 확신도에 비해 실제 정답률이 조금 낮은 편(보정 오차 ECE {cb['ece'] * 100:.1f}%p)이라, 확률을 그대로 믿기보다 기준값을 높게 잡았습니다. "
                  f"Scratch와 Loc처럼 어려운 패턴은 대부분 검토로 넘어갑니다.")}

# ---- 로트 SPC ----
spc = json.load(open(f'{O}/lot_spc.json'))
L = pd.read_csv(f'{O}/lot_table.csv', index_col=0)
roll = L.out.astype(int).rolling(400).sum(); end = int(roll.idxmax()); Z = L.loc[:end].tail(400)
gallery = json.load(open(f'{O}/lot_gallery.json'))
sh = spc['shift']; r0 = spc['runs'][0]
spc_d = {'lots': spc['lots_charted'], 'pbar': spc['pdbar'], 'out': spc['out_laney'], 'out_classic': spc['out_classic'],
         'conf_un': sh['unlabeled_conf98'], 'conf_lab': sh['labeled_conf98'],
         'top': spc['by_pattern'], 'runs': spc['runs'],
         'galInfo': {l: next(x for x in spc['by_pattern'] + spc['top10'] if x['lot'] == l) for l in gallery},
         'zoom': {'lot': Z.index.tolist(), 'name': Z.lot.tolist(), 'p': Z.pd.round(4).tolist(), 'ucl': Z.ucl.round(4).tolist(),
                  'out': Z.out.astype(int).tolist(), 'n': Z.n.astype(int).tolist()},
         'note': (f"전체 {spc['wafers']:,}장을 로트로 묶어 로트별 die 불량률을 관리도로 감시했습니다. 로트당 die가 수만 개라 일반 p 관리도는 한계가 지나치게 좁아 "
                  f"로트의 절반인 {spc['out_classic']:,}개를 이상으로 잡으므로, 로트 간 변동을 반영하는 Laney p' 관리도로 보정했습니다. "
                  f"가장 긴 이상 구간은 lot{r0['from']}~{r0['to']}의 연속 {r0['lots']}개 로트(die 불량률 {r0['die_fail'] * 100:.0f}%, 주로 {r0['dominant']})입니다. "
                  f"이상 로트의 패턴은 전문가 라벨과 확신도 98% 이상인 모델 판별만 썼습니다. 라벨 없는 웨이퍼는 불량이 더 적고 분포가 달라, "
                  f"확신도 98%를 넘는 비율이 {sh['unlabeled_conf98'] * 100:.0f}%(라벨 있는 웨이퍼 {sh['labeled_conf98'] * 100:.0f}%)에 그쳤기 때문입니다. "
                  f"로트 번호 순서가 실제 생산 시간 순서인지는 데이터에 없습니다.")}

data = {'classes': C, 'size': S, 'nets': nets,
        'samples': samples, 'calib': calib, 'spc': spc_d, 'gallery': gallery, 'modelDesc': a.desc,
        'metrics': {'acc': F['acc'], 'f1': F['macro_f1'], 'n_test': int(len(y)), 'n_train': F['n_train'],
                    'confusion': F['confusion'], 'models': models, 'note': a.note}}
html = open(a.template, encoding='utf-8').read().replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':'), ensure_ascii=False))
open(a.out, 'w', encoding='utf-8').write(html)
print('params', flat_size, 'samples', len(samples), 'html KB', len(html.encode()) // 1024)
