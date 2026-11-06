# 파이프라인 스크립트

노트북(`../notebooks/wafer_cnn.ipynb`)과 같은 작업을 단계별 스크립트로 나눈 것입니다. 실제 결과는 이 스크립트로 만들었습니다.

| 순서 | 스크립트 | 하는 일 |
|---|---|---|
| 1 | `prep2.py <라벨.pkl> <출력.npz> <크기> <nearest\|failmax>` | 크기 통일, none 1만 장 표본, 8:2 분할 |
| 2 | `train2.py <데이터.npz> aug <출력 접두어> <base\|deep>` | 소수 패턴 증강 후 CNN 학습·평가 (결과 .json, 확률 .npy, 가중치 .pt) |
| 3 | `gradcam.py`, `gradcam_fig.py` | Grad-CAM 계산과 그림, 불량 집중도 |
| 4 | `infer_all.py <LSWMD.pkl> <모델.pt> base nearest <출력.parquet>` | 811,457장 전체 추론 (512장씩 나눠 계산) |
| 5 | `lot_spc.py`, `lot_gallery.py` | 로트 die 불량률 Laney p' 관리도, 라벨 기반 p 관리도, 이상 로트 패턴 |
| 6 | `calibration.py` | 보정 곡선, 기준값별 자동/검토 분류 |
| 7 | `export_app2.py` + `app_template.html` | 모델 가중치를 넣은 판별 앱(`index.html`) 빌드 |
| - | `models.py` | 모델 구조(기본·깊은)와 크기 변환 함수 |
| - | `make_nb.py` | 노트북 생성 |

`LSWMD.pkl`은 구버전 pandas로 저장돼 있어, 옛 모듈 이름을 새 이름으로 연결하는 제한된 unpickler로 엽니다(`infer_all.py` 참고).
