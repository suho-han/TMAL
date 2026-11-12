# Config YAML Guide

`scripts/train_launcher_from_config.py`로 실행하는 YAML 설정 파일 작성 가이드.

```bash
python scripts/train_launcher_from_config.py --config scripts/configs/<name>.yaml [--device N] [--dry_run] [--smoke]
```

---

## 최소 구조

```yaml
# single: 한 데이터셋, 한 구성
dataset: drive
mode: single
device: 0
single:
  conn_num: 8
  label_mode: binary

# multi: 여러 데이터셋 + 옵션 조합 grid sweep
dataset: [drive, chase]
mode: multi
device: 0
multi:
  epochs: 500
  folds: 1
  conn_nums: [8, 24]
  label_modes: [binary, dist, dist_inverted]
  dist_aux_losses: [smooth_l1, gjml_sf_l1]
```

---

## 최상위 키

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `dataset` | string \| list | 필수 | 데이터셋. multi 모드에서 리스트 가능 |
| `mode` | string | 필수 | `single` \| `multi` |
| `device` | int | `0` | CUDA device 번호 |
| `dist_sf_l1_gamma` | float | `1.0` | `gjml_sf_l1` loss의 gamma |
| `batch_size` | int | preset 기본값 | preset 배치 크기 덮어쓰기 |
| `output_dir` | string | `"output"` | 결과 저장 최상위 디렉토리 |
| `skip_completed` | bool | deprecated | 기존 호환용 키. 현재는 무시되며 완료 실험은 항상 자동으로 건너뜀 |
| `test_only` | bool | `false` | 평가만 실행 |
| `pretrained` | string | — | `test_only` 시 체크포인트 경로 |

### 지원 데이터셋 및 preset

| `dataset` | resize | batch_size | 비고 |
|-----------|--------|------------|------|
| `chase` | 960×960 | 4 | |
| `cremi` | 256×256 | 4 | |
| `drive` | 512×512 | 4 | |
| `deepglobe` | 1024×1024 | 2 | split root: `data/DeepGlobe/splitted` |
| `isic2018` | 224×320 | 64 | `prepare_isic2018_npy.py` 사전 실행 필요 |
| `octa500` | 512×512 | 16 | `octa_variant: 3M \| 6M` 필요 |

---

## single 블록 키

single.final_decision은 max, mean, weighted_mean 중 하나이며 기본값은 max입니다.

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `conn_num` | int | `8` | `8` \| `24` |
| `conn_layout` | string | 자동 | `standard8`(기본, conn_num=8) \| `out8`(conn_num=8) \| `full24`(conn_num=24) |
| `label_mode` | string | `binary` | `binary` \| `dist` \| `dist_inverted` |ㄱ
| `dist_aux_loss` | string | `smooth_l1` | `smooth_l1` \| `gjml_sf_l1` \| `cl_dice` |
| `folds` | int | `1` | K-fold 수 |
| `epochs` | int | 자동 결정 | 생략 시 아래 규칙 적용 |
| `octa_variant` | string | `6M` | dataset=octa500일 때만 유효 |

**epochs 자동 결정 규칙** (single):

- `isic2018` → 500
- `dist` / `dist_inverted` → 260
- `conn_num=24`, binary → 390
- `conn_num=8`, binary → 130

---

## multi 블록 키

multi.final_decisions는 final_decision sweep 목록입니다. 기본값은 [max]입니다.

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `conn_nums` | list\[int\] | `[8, 24]` | sweep할 conn_num 목록 |
| `conn_layouts` | list\[string\] | `[null]` | sweep할 conn_layout 목록 |
| `label_modes` | list\[string\] | `[binary, dist, dist_inverted]` | sweep할 label_mode |
| `dist_aux_losses` | list\[string\] | `[gjml_sf_l1, smooth_l1]` | dist/dist_inverted 전용 |
| `binary_dist_aux_loss` | string | `smooth_l1` | binary label_mode 전용 aux loss |
| `octa_variants` | list\[string\] | `[3M, 6M]` | dataset=octa500일 때만 유효 |
| `use_seg_aux` | bool | `false` | grid run에 SegAux 활성화 |
| `seg_aux_weights` | list\[float\] | `[null]` | `use_seg_aux=true`이면 필수. sweep할 SegAux weight 목록 (단수 `seg_aux_weight` 별칭 허용) |

| `conn_aux_c3_weights` | list\[float\] | `[0.3]` | dg/dg_direct에서 inner(`i`) 보조 loss 가중치 sweep (단수 `conn_aux_c3_weight` 별칭) |
| `conn_aux_c4_weights` | list\[float\] | `[0.2]` | dg/dg_direct에서 middle(`m`) 보조 loss 가중치 sweep (단수 `conn_aux_c4_weight` 별칭) |
| `conn_aux_c5_weights` | list\[float\] | `[0.2]` | dg/dg_direct에서 outer(`o`) 보조 loss 가중치 sweep (단수 `conn_aux_c5_weight` 별칭) |
| `experiment_name_suffix` | string | `""` | grid run의 모든 출력 폴더명에 동일 접미사 부여 |
| `epochs` | int | `500` | |
| `folds` | int | `1` | |
| `experiments_only` | bool | `false` | grid 없이 experiments만 실행 |

---

## 완료 실험 자동 스킵

- launcher는 생성된 run마다 완료 산출물 존재 여부를 검사합니다.
- 아래 중 하나가 있으면 이미 완료된 실험으로 간주하고 경고 후 실행을 건너뜁니다.
  - `<output_dir>/<dataset>/<experiment_name>/final_results_<fold>.csv`
  - `<output_dir>/<dataset>/<experiment_name>/models/<epochs>_model.pth`
- 건너뛴 재시도 run은 `output/retried_exp.md`에 append 방식으로 기록됩니다.
- `skip_completed`, `single.skip_completed`, `multi.skip_completed`는 deprecated이며 현재 동작을 바꾸지 않습니다.

---

## final_decision

Bilateral_voting의 방향 reduction은 final_decision으로 선택합니다.

- max는 upstream-compatible 기본값이며 기존 experiment name을 유지합니다.
- mean은 8방향 평균입니다.
- weighted_mean은 직선 방향 weight 1, 대각 방향 weight 0.5의 정규화 weighted average입니다.
- mean과 weighted_mean은 standard8 fused/main path에서 사용합니다. full24/out8 direct path는 거부하며, io의 non-standard auxiliary branch는 기존 max를 유지합니다.
- 비기본 decision의 출력 폴더에는 fd_mean 또는 fd_weighted_mean 식별자가 붙습니다.

## Proposed fd comparison config

comparison_models_final_decision.yaml은 summary의 Proposed 조합을 고정하고 final_decisions: [mean, weighted_mean]을 전체 comparison dataset에 대해 실행합니다. 기본 anchor인 final_decision=max은 paper table query에서 별도로 유지됩니다.

## conn_fusion 옵션

`conn_fusion != none`이면 **`conn_num=8`, `conn_layout=standard8`, `num_class=1` 필수**.

| 값 | 설명 |
|----|------|
| `none` | 기본 경로 |
| `gate` | 게이트 기반 퓨전 |
| `scaled_sum` | 잔차 스케일 합산 |
| `conv_residual` | 컨볼루션 잔차 퓨전 |
| `dg` | Decoder-Guided (DGRF) |
| `dg_direct` | DGRF direct (SegAux 없이도 실행 가능) |

### fusion 브랜치 (`fusion_branches`)

문자열 플래그를 조합해 활성 브랜치 집합을 지정. `dg`, `dg_direct`에서만 의미가 있음.
기본값 `io`는 inner(3×3) + outer(5×5) 두 브랜치를 켜는 의미.

| 문자 | 브랜치 | 커널/오프셋 | 모델 출력 키 | 대응 aux 가중치 |
|------|--------|------------|--------------|-----------------|
| `i` | inner | 3×3 (`standard8`) | `inner` | `conn_aux_c3_weight` |
| `m` | middle (`mid8`) | 5×5 (대각 제외) | `middle` | `conn_aux_c4_weight` |
| `o` | outer (`out8`) | 5×5 외곽 | `outer` | `conn_aux_c5_weight` |
| `p` | scale3 (`scale3_8`) | 7×7 외곽 | `scale3` | `conn_aux_c6_weight` |
| `f` | scale4 (`scale4_8`) | 9×9 외곽 | `scale4` | `conn_aux_c7_weight` |

예: `fusion_branches: imof` → inner + middle + outer + scale4 4개 브랜치 활성.

**중심 픽셀 `V`를 기준으로 한 각 브랜치 오프셋 위치 (`.`은 어느 브랜치에도 속하지 않는 위치):**

i + m + o만 보이는 5×5 영역:

```text
o  m  o  m  o
m  i  i  i  m
o  i  V  i  o
m  i  i  i  m
o  m  o  m  o
```

i + m + o + p + f 모두를 포함하는 9×9 전체 영역:

```text
f  .  .  .  f  .  .  .  f
.  p  .  .  p  .  .  p  .
.  .  o  m  o  m  o  .  .
.  .  m  i  i  i  m  .  .
f  p  o  i  V  i  o  p  f
.  .  m  i  i  i  m  .  .
.  .  o  m  o  m  o  .  .
.  p  .  .  p  .  .  p  .
f  .  .  .  f  .  .  .  f
```

- 각 브랜치는 항상 정확히 **8개 오프셋**(채널 수 8)을 사용. 위 그림에서 각 글자가 8번씩 등장.
- `i`: 거리 1의 8-이웃 (3×3 직접 이웃).
- `m`: 거리 √5의 "knight-move" 8개 — (±2, ±1) / (±1, ±2).
- `o`: 거리 2의 축·대각 8개 — (±2, 0) / (0, ±2) / (±2, ±2).
- `p`: 거리 3의 축·대각 8개 — (±3, 0) / (0, ±3) / (±3, ±3).
- `f`: 거리 4의 축·대각 8개 — (±4, 0) / (0, ±4) / (±4, ±4).

### fusion 파라미터

| 키 | 기본값 | 적용 대상 (어떤 conn_fusion / 브랜치) |
|----|--------|---------------------------------------|
| `fusion_lambda_fused` | `0.3` | 모든 fusion. **fused 브랜치** affinity 가중치 |
| `fusion_residual_scale` | `0.2` | `scaled_sum` 전용. fused = main + scale·residual |
| `fusion_gate_reg_weight` | `0.01` | `dg`, `dg_direct`. fusion gate L1 정규화 가중치 |
| `conn_aux_c3_weight` | `0.3` | `dg`, `dg_direct`. **inner 브랜치(`i`)** aux loss 가중치 |
| `conn_aux_c4_weight` | `0.2` | `dg`, `dg_direct`. **middle 브랜치(`m`)** aux loss 가중치 (브랜치 활성 시) |
| `conn_aux_c5_weight` | `0.2` | `dg`, `dg_direct`. **outer 브랜치(`o`)** aux loss 가중치 (브랜치 활성 시) |
| `conn_aux_c6_weight` | `0.3` | `dg`, `dg_direct`. **scale3 브랜치(`p`)** aux loss 가중치 (브랜치 활성 시) |
| `conn_aux_c7_weight` | `0.2` | `dg`, `dg_direct`. **scale4 브랜치(`f`)** aux loss 가중치 (브랜치 활성 시) |

> `fusion_loss_profile` 및 관련 옵션(`fusion_lambda_inner`, `fusion_lambda_outer`)은 더 이상 지원되지 않습니다. 손실은 fused 분기(fused)와, `dg`/`dg_direct`의 보조 경로(`conn_aux_c*`)를 통해만 구성됩니다.

### multi에서 fusion sweep 방법

**방법 1 — 단순 리스트**: `conn_fusions`, `fusion_residual_scales` 키 사용.

**방법 2 — fusion_matrix**: conn_fusion별로 옵션을 개별 제어. `fusion_matrix`가 있으면 `conn_fusions` 무시.

```yaml
multi:
  fusion_matrix:
    - conn_fusion: none
    - conn_fusion: gate
    - conn_fusion: scaled_sum
      fusion_residual_scales: [0.1, 0.2, 0.3]  # scaled_sum만 여러 값 허용
    - conn_fusion: dg
```

---

## SegAux

`use_seg_aux: true`이면 `seg_aux_weight` 반드시 지정. `dg_direct`는 SegAux 없이도 실행 가능하지만, 실제로는 `use_seg_aux: true`로 쓰는 것이 일반적.

---

## experiments (개별 실험 추가)

각 multi.experiments 항목은 final_decision을 개별적으로 override할 수 있습니다.

grid sweep 이후 특정 구성을 추가 실행. `experiments_only: true`이면 grid 없이 experiments만 실행.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `datasets` / `dataset` | top-level dataset 전체 | 실행할 데이터셋 (부분집합만 허용) |
| `conn_num` | `conn_nums[0]` | |
| `label_mode` | `label_modes[0]` | |
| `dist_aux_loss` | label_mode에 따라 자동 | |
| `conn_fusion` | `none` | |

| `use_seg_aux` | `false` | |
| `seg_aux_weight` | — | `use_seg_aux=true`이면 필수 |
| `output_dir` | 상위 `output_dir` 상속 | 실험별 결과 루트 override |
| `experiment_name_suffix` | 없음 | 같은 output root 아래에서 폴더명만 추가로 구분할 때 사용 |
| `epochs` | `multi.epochs` 상속 | |
| `device` | top-level device 상속 | |

---

## 학습 제어

`single` 또는 `multi` 블록 안에 작성.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `monitor_metric` | `val_dice` | `val_dice` \| `val_loss` |
| `early_stopping_patience` | `20` | 0이면 비활성화 |
| `early_stopping_min_delta` | `0.001` | |
| `early_stopping_tie_eps` | `0.0001` | val_dice 동점 판정 기준 |
| `early_stopping_stop_interval` | `10` | 이 epoch 배수에서만 실제 중지 |
| `tie_break_with_loss` | `true` | val_dice 동점 시 val_loss로 tiebreak |
| `save_best_only` | `false` | true이면 주기 체크포인트 비활성화 |

---

## 출력 디렉토리 명명 규칙

`output_dir/<dataset>/<experiment_name>/`에 저장. `<experiment_name>` 패턴:

| 조건 | 이름 예시 |
|------|-----------|
| binary, fusion=none | `binary_8_bce` |
| binary, fusion=dg, profile=A | `binary_dg_A_8_bce` |
| dist, fusion=none | `dist_8_smooth_l1` |
| scaled_sum, rs=0.2 | `binary_scaled_sum_A_rs0.2_8_bce` |
| dg_direct, fusion_branches=io (기본) | `dist_inverted_dg_direct_A_8_smooth_l1` |
| dg_direct, fusion_branches=im (비기본) | `dist_inverted_dg_direct_A_im_8_smooth_l1` |
| conn_layout=out8 | `binary_8_out8_bce` |
| use_seg_aux, weight=0.5 | `..._segaux_w0.5` 접미사 |
| experiment_name_suffix=io_modified | `..._io_modified` 접미사 |
| multi sweep: `fusion_lambda_inners=[0.1,0.3]` 등 | sweep된 키별로 `li{값}`/`lo{값}`/`c3w{값}`/`c4w{값}`/`c5w{값}` 자동 접미사로 출력 폴더 구분 (sweep 길이=1이면 표시 안 함) |

---

## CLI 오버라이드

| 옵션 | 설명 |
|------|------|
| `--device N` | config의 device 덮어쓰기 |
| `--batch-size N` | config의 batch_size 덮어쓰기 |
| `--dry_run` | 명령어만 출력, 실행 안 함 |
| `--smoke` | 기본적으로 YAML에서 생성된 전체 실험을 5 epoch, batch=1로 `output_smoke/`에 저장 |
| `--smoke_limit N` | smoke 시 첫 N개 실험만 실행하도록 제한 |
| `--test_only` | 평가만 실행 |
| `--pretrained PATH` | 체크포인트 경로 (생략 시 자동 탐색) |

### 다른 서버에서 실행할 때

- `device` 값은 현재 서버의 GPU 번호를 기본값으로 적어둘 수 있지만, 다른 서버에서는 `--device N`으로 덮어쓰는 것이 안전합니다.
- 예시:

```bash
scripts/train_launcher.sh --config scripts/configs/<name>.yaml --device 0
```

- 서버마다 GPU 점유 상태가 다르므로, config의 `device`를 고정값으로 믿기보다 런처 오버라이드를 우선 사용하세요.
