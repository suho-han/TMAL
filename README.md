# TMAL: Topology-preserving Multi-scale Affinity Learning for Curvilinear Structure Segmentation

Official implementation of TMAL for the WACV 2027 submission.

![figure](images/figure_1.png)

---

## Abstract

Curvilinear structures, such as retinal vessels and surface cracks, require pixel-level accuracy and structural continuity. Pixel-wise supervision does not model connections between pixels, and topology-based losses still output conventional masks. Connectivity-based methods explicitly predict such connections, but their binary labels are limited to a fixed local range. These labels assign the same target to thin segments and thicker foreground regions and cannot express larger-offset connections. To address these limitations, we propose **Topology-preserving Multi-offset Affinity Learning (TMAL)**. TMAL uses soft affinities to emphasize thin, boundary-adjacent regions and predicts directional affinities at multiple offsets to capture wider connectivity. It combines these affinities with weights derived from decoder features, so that local continuity and wider spatial context are used together. An auxiliary segmentation branch adds region-level supervision to maintain overlap accuracy. On the retinal vessel dataset DRIVE and the crack inspection dataset DeepCrack, TMAL achieves the highest Dice, IoU, and clDice scores and substantially reduces the connected-component count error $\Delta\beta_0$.

---

## Key Contributions

- Replaces the binary connectivity label with a continuous, distance-aware directional affinity target that reflects structure thickness and the spatially varying uncertainty of a connection, concentrating supervision where disconnections actually occur.
- Extends affinity learning across spatial scales (MAP) and fuses them adaptively through decoder-conditioned gates (AF), capturing local continuity and extended-neighborhood connectivity together.
- Improves both segmentation accuracy and topology preservation on a medical and an industrial benchmark, leading in Dice, IoU, clDice and Betti-0 error on both.

---

## Datasets

TMAL is evaluated on two curvilinear-structure benchmarks from different domains:

- **DRIVE** — 40 colour fundus images (565×584) with binary retinal vessel masks, official 20/20 train/test split.
- **DeepCrack** — 537 concrete-surface images (544×384) with crack annotations, official 400/137 train/test split.

### Download & layout

**DRIVE** — download from the official [Grand Challenge page](https://drive.grand-challenge.org/) (registration required) and arrange it as:

```Plain Text
data/DRIVE/
  train/images/*.tif   train/masks/*.**gif**
  test/images/*.tif    test/masks/*.gif
```

**DeepCrack** — download from the dataset authors' repo, [yhlleo/DeepCrack](https://github.com/yhlleo/DeepCrack) (Liu et al., *Neurocomputing* 2019; see its README for the current download link), and arrange it as:

```Plain Text
data/DeepCrack/
  train_img/*.jpg   train_lab/*.png
  test_img/*.jpg    test_lab/*.png
```

Both `data_loader/GetDataset_DRIVE.py` and `data_loader/GetDataset_DeepCrack.py` expect exactly these layouts under `train_root` (i.e. `data/DRIVE`, `data/DeepCrack`).

`label_mode=dist`/`dist_inverted` additionally need precomputed distance-map labels next to the binary masks (`masks_dist(_inverted)/*.npy` for DRIVE, `train_lab_dist(_inverted)/*.npy` / `test_lab_dist(_inverted)/*.npy` for DeepCrack). Generate them from the binary masks with:

```bash
scripts/build_dist_maps.py --mask_dir data/DRIVE/train/masks --mask_glob "*.gif"
scripts/build_dist_maps.py --mask_dir data/DRIVE/test/masks --mask_glob "*.gif"
scripts/build_dist_maps.py --mask_dir data/DeepCrack/train_lab --mask_glob "*.png"
scripts/build_dist_maps.py --mask_dir data/DeepCrack/test_lab --mask_glob "*.png"
```

(`binary` label mode needs no extra step — it reads the masks directly.)

---

## Results

Mean over the test set; TMAL attains the best Dice, IoU and clDice on both benchmarks and the lowest Betti-0 error against mask-based, topology-based, and connectivity-based baselines. **Bold** = best, *italic* = second best.

### DRIVE

| Category           | Method     | Dice ↑     | IoU ↑      | clDice ↑   | β0 ↓        | β1 ↓       | HD95 ↓     |
|--------------------|------------|------------|------------|------------|-------------|------------|------------|
| Mask-based         | U-Net      | 0.7813     | 0.6415     | 0.7818     | 446.15      | 6.20       | 9.8495     |
| Mask-based         | FR-UNet    | 0.7573     | 0.6097     | 0.7716     | 484.65      | 7.20       | 10.5546    |
| Mask-based         | CS-Net     | 0.7556     | 0.6076     | 0.7611     | 454.65      | 5.85       | 11.4094    |
| Mask-based         | VGAN       | *0.8167*   | *0.6905*   | *0.8168*   | 300.65      | 5.50       | 7.4352     |
| Mask-based         | SegDiff    | 0.7186     | 0.5617     | 0.6998     | 369.65      | 9.30       | 11.2767    |
| Mask-based         | MedSegDiff | 0.7569     | 0.6092     | 0.7168     | *204.45*    | **4.95**   | **2.5895** |
| Topology-based     | clDice     | 0.6561     | 0.4888     | 0.8159     | 513.30      | *5.20*     | 9.0765     |
| Topology-based     | CAPE       | 0.7097     | 0.5503     | 0.7570     | 307.10      | 6.75       | *3.9684*   |
| Connectivity-based | DconnNet   | 0.7762     | 0.6346     | 0.7835     | 452.80      | 5.35       | 9.4818     |
| **TMAL (Ours)**    |            | **0.8295** | **0.7090** | **0.8773** | **24.50**   | 20.95      | 8.4776     |

### DeepCrack

| Category           | Method     | Dice ↑     | IoU ↑      | clDice ↑   | β0 ↓       | β1 ↓       | HD95 ↓     |
|--------------------|------------|------------|------------|------------|------------|------------|------------|
| Mask-based         | U-Net      | 0.7931     | 0.6814     | 0.8586     | 4.9283     | 1.3122     | 30.6565    |
| Mask-based         | FR-UNet    | 0.6748     | 0.5593     | 0.7366     | 5.3333     | *0.9283*   | 61.1240    |
| Mask-based         | CS-Net     | 0.8051     | 0.6945     | 0.8721     | 5.4177     | 1.4304     | 32.7916    |
| Mask-based         | VGAN       | 0.6249     | 0.4933     | 0.7021     | 5.7384     | 1.5443     | 52.2659    |
| Mask-based         | SegDiff    | 0.7900     | 0.6729     | 0.8624     | 5.6456     | 1.5316     | **14.4408**|
| Mask-based         | MedSegDiff | 0.7749     | 0.6452     | 0.8797     | 4.6203     | 1.7046     | 47.8944    |
| Topology-based     | clDice     | 0.7673     | 0.6642     | 0.8258     | *4.2152*   | 2.1055     | 48.9162    |
| Topology-based     | CAPE       | 0.8049     | 0.6896     | 0.8698     | 7.0802     | 4.2700     | *19.9207*  |
| Connectivity-based | DconnNet   | *0.8252*   | *0.7134*   | *0.8927*   | 4.5148     | 1.0506     | 26.3769    |
| **TMAL (Ours)**    |            | **0.8439** | **0.7427** | **0.9074** | **1.9831** | **0.8776** | 29.1717    |

Full comparisons and the ablation study isolating each module's contribution are in the paper.

---

## Modules

- **ATC** — Affinity Target Conversion: build distance-aware affinity targets.
- **MAP** — Multi-scale Affinity Prediction: predict directional affinities at multiple scales.
- **AF** — Affinity Fusion: fuse multi-scale affinity predictions at the logit stage.
- **DMD** — Distance Map Decoding: decode the fused affinities into a distance map.
- **AuxSeg** — Auxiliary Segmentation branch preserving pixel-level accuracy.

---

## Repository layout

```Plain Text
src/            Core implementation (models, losses, engine, data)
  model/        Model architectures (TMAL + comparison baselines)
  losses/       Loss functions
  engine/       Training / validation / test loops
  data/         Dataset builders
metrics/        Evaluation metrics (clDice, Betti errors, HD95, ...)
data_loader/    Dataset loaders (DRIVE, DeepCrack, ...)
```

---

## Installation

```bash
git clone https://github.com/tmal-wacv2027/TMAL.git
cd TMAL
```

Requires Python >= 3.11 and a CUDA-enabled PyTorch build. Set up either directly on the host, or in Docker.

### Local (uv)

```bash
uv sync            # resolves dependencies from pyproject.toml
```

The pinned PyTorch build uses the CUDA 12.8 wheel index
(`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`).

To restrict training/testing to a single GPU, `train.py`'s `--device` selects the CUDA device index directly (it defaults to `1`); `CUDA_VISIBLE_DEVICES` additionally hides the other GPUs from the process:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_ours.sh drive train --device 0
```

### Docker

The `Dockerfile` builds the same environment via `uv sync`. `torch==2.11.0+cu128` ships its own CUDA runtime, so the image needs no CUDA toolkit base — just the NVIDIA driver and the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) on the host so `docker run --gpus` can pass a GPU through.

```bash
docker build -t tmal .

docker run --gpus all -it \
  -v $(pwd)/data:/app/data -v $(pwd)/ckpts:/app/ckpts \
  tmal bash
```

`data/` and `ckpts/` are gitignored and not baked into the image — bind-mount them as above.

To restrict training/testing to a single GPU, limit which device the container can see and pass `--device` to `train.py`:

```bash
docker run --gpus '"device=0"' -it \
  -v $(pwd)/data:/app/data -v $(pwd)/ckpts:/app/ckpts \
  tmal bash
# inside the container:
scripts/run_ours.sh drive train --device 0
```

---

## Training & Inference

`scripts/run_ours.sh` builds the `train.py` command from `scripts/configs/{drive,deepcrack}_tmal_test.yaml`
(the TMAL (Ours) hyperparameters) and runs it:

```bash
scripts/run_ours.sh <drive|deepcrack> <train|test> [extra train.py flags]
```

`test` runs test-only against the shipped checkpoint (`ckpts/drive_model.pth` /
`ckpts/deepcrack_model.pth`); trailing flags override the yaml (e.g. `--device 1`); `--dry_run`
just prints the command. For new experiments, use `scripts/train_launcher.sh` with a
`single`/`multi` YAML instead — see `scripts/train_launcher.md`.

Each run writes its outputs under `output/<dataset>/<experiment_name>/`, including a
`result.txt` summary in the paper's Dice/IoU/clDice/β0/β1/HD95 format (mean ± std over the
evaluated split) — check it first for a quick read of a run's results, alongside the more
detailed `results.csv` / `final_results.csv` and `models/test_sample_metrics.csv`.

---

## Acknowledgments

TMAL builds on and is grateful to the following prior work:

- **DConnNet** — Zyun-Y/DconnNet, *"Directional Connectivity-based Segmentation of Medical Images"*, CVPR 2023. TMAL's codebase (data pipeline, directional connectivity utilities) is derived from this repository. <https://github.com/Zyun-Y/DconnNet>
- **BiconNet** — Yang et al., *"BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection"*, Pattern Recognition, 2022. The Bilateral Voting formulation adapted in TMAL's Distance Map Decoding (Bilateral Affinity Calculation) originates from this work. <https://github.com/Zyun-Y/BiconNets>
