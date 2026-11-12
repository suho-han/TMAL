# `scripts/train_launcher.sh` 사용 방법

## 1) 기본 실행 형식

```bash
scripts/train_launcher.sh --config <yaml 경로> [--device GPU_ID] [--batch-size N] [--dry_run] [--test_only] [--pretrained [PATH]] [--smoke] [--smoke_limit N]
```

- `--config`: 실행 스펙 YAML 파일 경로 (필수)
- `--device`: YAML의 `device` 값을 CLI에서 덮어쓰기 (선택)
- `--batch-size`: YAML의 `batch_size` 값을 CLI에서 덮어쓰기 (선택)
- `--dry_run`: 실제 실행 없이 생성될 `train.py` 명령만 출력 (선택)
- `--smoke`: 최소 설정으로 smoke run 실행(기본: YAML에서 생성된 전체 run, 각 run은 5 epoch, batch_size=1, target_fold=1, output_dir=output_smoke, 학습/검증/최종평가 모두 1배치 제한) (선택)
- `--smoke_limit N`: `--smoke` 시 첫 N개 run만 실행하도록 제한 (선택)
- `--test_only`: YAML과 무관하게 test-only 실행으로 강제 (선택)
- `--pretrained [PATH]`: test-only 체크포인트 경로 오버라이드 (선택)
- launcher는 이미 완료된 실험을 자동으로 재학습하지 않으며, 경고 후 건너뜁니다.
- 건너뛴 재시도 실험은 `output/retried_exp.md`에 누적 기록합니다.

## 1.5) 다른 서버로 넘길 때

- GPU가 부족한 현재 서버에서 실행을 미루고 다른 서버로 넘길 때는, YAML은 그대로 두고 `--device N`만 대상 서버에 맞게 바꾸는 것이 가장 안전합니다.
- OCTA fine sweep처럼 장시간 도는 작업은 아래처럼 실행하면 됩니다.

```bash
scripts/train_launcher.sh --config scripts/configs/<name>.yaml --device 0
```

- 다른 서버에서 device 번호가 다르면 `0` 대신 실제 사용 가능한 번호로 바꾸세요.

## final_decision

launcher YAML에서 max, mean, weighted_mean 중 하나를 지정할 수 있습니다. mean과 weighted_mean은 standard8에서만 지원하고, non-default 실행 폴더에는 fd 식별자가 추가됩니다.

## 2) 사전 조건

- 프로젝트 루트에서 실행
- `.venv` 환경 사용
- `PyYAML` 설치 필요

## 3) 핵심 예시

```bash
scripts/train_launcher.sh --config scripts/configs/<name>.yaml --dry_run
scripts/train_launcher.sh --config scripts/configs/<name>.yaml --smoke
scripts/train_launcher.sh --config scripts/configs/<name>.yaml --device 1
```

`<name>.yaml`은 아래 4)의 `single` / `multi` 스키마로 직접 작성한 launcher 입력 YAML을 가리킵니다.
`scripts/configs/drive_tmal_test.yaml`, `deepcrack_tmal_test.yaml`은 launcher 입력이 아니라
논문에 보고된 TMAL(Ours) 결과를 재현하는 정확한 하이퍼파라미터 기록(resolved config dump)이며,
`train.py` 직접 호출로만 사용합니다 — 실행 방법은 리포지토리 루트 `README.md`의
"Training & Inference" 절을 참고하세요.

## 4) YAML 요약

### single

```yaml
dataset: chase
mode: single
device: 3
single:
  conn_num: 8
  label_mode: binary
  dist_aux_loss: smooth_l1
  folds: 1
  epochs: 130
```

### multi

```yaml
dataset: [chase, drive]
mode: multi
device: 3
multi:
  epochs: 500
  folds: 1
  conn_nums: [8, 24]
  label_modes: [binary, dist, dist_inverted]
  dist_aux_losses: [smooth_l1, gjml_sf_l1]
  # explicit run list (optional)
  experiments_only: false
  experiments:
    - datasets: [drive]   # optional override; defaults to top-level dataset list
      conn_num: 8
      conn_layout: standard8
      label_mode: dist
      dist_aux_loss: smooth_l1
      conn_fusion: none
  # fork-specific fusion ablations
  conn_fusions: [none, gate, scaled_sum, conv_residual]
  fusion_lambda_fused: 0.3
  fusion_residual_scales: [0.1, 0.2, 0.3, 0.5]
```

## 5) test-only

- `single.test_only: true` 또는 `multi.test_only: true` 지원
- `pretrained` 미지정 시 자동 경로:
  - `<output_dir>/<dataset>/<experiment_name>/models/best_model.pth`
- `pretrained` 포맷 문자열 지원 키:
  - `{dataset}`, `{conn_num}`, `{label_mode}`, `{dist_aux_loss}`, `{conn_layout}`, `{conn_fusion}`, `{experiment_name}`
- 각 학습 run은 launcher 원본 YAML 복사 대신, 실제 실행 인자를 기준으로 한 실험별 `config.yaml`을 결과 디렉토리에 저장합니다.

## 6) 주의 사항

- 완료 판정 기준:
  - `<output_dir>/<dataset>/<experiment_name>/final_results_<fold>.csv`
  - 또는 `<output_dir>/<dataset>/<experiment_name>/models/<epochs>_model.pth`
- 완료 산출물이 있으면 `--dry_run`, `--smoke` 포함 모든 launcher 실행에서 해당 run을 경고 후 스킵합니다.
- 기존 `skip_completed`, `single.skip_completed`, `multi.skip_completed` 키는 deprecated이며, 현재는 모두 무시됩니다.
- `direction_grouping`, `direction_fusion` 키는 제거되어 더 이상 지원되지 않습니다.
- `decoder_fusion`, `decoder_fusions`, `lambda_vote_aux` 키는 제거되어 더 이상 지원되지 않습니다.
- `conn_fusion != none`은 현재 single-class/`conn_num=8`/`conn_layout=standard8` 조합만 지원합니다.
- RETOUCH 계열은 현재 launcher에서 지원하지 않습니다.
- `decoder_guided`는 launcher 입력에서 legacy alias로 허용되지만, 새 canonical 이름은 `dg`입니다.
- `dg_direct`는 `SegAux`를 켜지 않아도 허용되며, `SegAux`를 켠 경우 mask head에 `final_feat`만 사용합니다.
- `scripts/configs/`는 현재 `drive_tmal_test.yaml` / `deepcrack_tmal_test.yaml`(TMAL(Ours) 하이퍼파라미터 기록)과 `guide.md`만 유지합니다. 새 grid-sweep 실험은 4)의 스키마로 새 YAML을 만들어 실행하세요.
