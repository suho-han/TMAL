# `scripts/eta_monitor.py` 사용 방법

## 1) 목적

`results.csv`의 `epoch`, `elapsed_hms`를 이용해 학습 ETA(예상 종료 시간)를 계산합니다.

## 2) 기본 실행 형식

```bash
.venv/bin/python scripts/eta_monitor.py --csv <results.csv 경로> [--total-epochs 500] [--watch] [--interval 10] [--tz Asia/Seoul]
```

- `--csv`: 대상 `results.csv` 경로 (필수)
- `--total-epochs`: 전체 epoch 수 (기본 `500`)
- `--watch`: 주기적으로 ETA 갱신 출력
- `--interval`: watch 갱신 주기(초), 기본 `10`
- `--tz`: 예상 종료 시각 출력 타임존, 기본 `Asia/Seoul`

## 3) 예시

### 1회 ETA 계산

```bash
.venv/bin/python scripts/eta_monitor.py \
  --csv output/octa500-3M/dist_inverted_24_gjml_sf_l1/results.csv
```

### 주기 모니터링(5초 간격)

```bash
.venv/bin/python scripts/eta_monitor.py \
  --csv output/octa500-3M/dist_inverted_24_gjml_sf_l1/results.csv \
  --watch --interval 5
```

### 총 epoch 덮어쓰기

```bash
.venv/bin/python scripts/eta_monitor.py \
  --csv output/isic/dist_24_gjml_sf_l1/results.csv \
  --total-epochs 260
```

## 4) 출력 해석

- `progress=<last_epoch>/<total_epochs>`
- `avg_sec_per_epoch`: 전체 epoch 평균 시간(초)
- `eta_duration`: 남은 예상 소요 시간
- `expected_finish_time`: 예상 종료 시각(`--tz` 기준)
- `status`:
  - `running`: `last_epoch < total_epochs`
  - `completed_or_overrun`: `last_epoch >= total_epochs`

## 5) 에러 조건

- CSV 파일 없음
- CSV에 `epoch`, `elapsed_hms` 컬럼 없음
- 파싱 가능한 유효 row가 없음
- 잘못된 타임존/잘못된 인자(`--total-epochs <= 0`, `--interval <= 0`)
