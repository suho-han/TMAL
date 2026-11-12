import argparse
import os
import random
import sys
from pathlib import Path

import autorootcwd

_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import torch
import yaml

from connect_loss import (
    is_default_connectivity_layout,
    normalize_conn_layout,
    normalize_final_decision,
    resolve_connectivity_layout,
)
from model.DconnNet import normalize_conn_fusion_mode
from model.registry import build_model as _build_model
from solver import Solver
from data import build_dataloaders, build_datasets
from utils import create_exp_directory


def get_experiment_output_name(args):
    layout_suffix = ''
    if not is_default_connectivity_layout(args.conn_num, args.conn_layout):
        layout_suffix = f'_{args.conn_layout}'

    conn_fusion = normalize_conn_fusion_mode(getattr(args, 'conn_fusion', 'none'))
    if conn_fusion == 'none':
        if args.label_mode == 'binary':
            base_name = f'binary_{args.conn_num}{layout_suffix}_bce'
        else:
            base_name = f"{args.label_mode}_{args.conn_num}{layout_suffix}_{args.dist_aux_loss}"
    else:
        fusion_tag = conn_fusion
        if conn_fusion == 'scaled_sum':
            residual_scale = float(getattr(args, 'fusion_residual_scale', 0.2))
            scale_str = f"{residual_scale:.6f}".rstrip("0").rstrip(".")
            fusion_tag = f'{fusion_tag}_rs{scale_str}'
        fusion_branches = str(getattr(args, 'fusion_branches', 'none') or 'none').strip().lower()
        if fusion_branches and fusion_branches != 'none':
            fusion_tag = f'{fusion_tag}_{fusion_branches}'
        if args.label_mode == 'binary':
            base_name = f'binary_{fusion_tag}_{args.conn_num}{layout_suffix}_bce'
        else:
            base_name = f"{args.label_mode}_{fusion_tag}_{args.conn_num}{layout_suffix}_{args.dist_aux_loss}"

    if getattr(args, 'use_seg_aux', False):
        weight = getattr(args, 'seg_aux_weight', None)
        if weight is not None:
            base_name = f"{base_name}_segaux_w{weight}"
        else:
            base_name = f"{base_name}_segaux"

    final_decision = normalize_final_decision(getattr(args, 'final_decision', 'max'))
    if final_decision != 'max':
        base_name = f"{base_name}_fd_{final_decision}"

    experiment_name_suffix = str(getattr(args, 'experiment_name_suffix', '') or '').strip()
    if experiment_name_suffix:
        base_name = f"{base_name}_{experiment_name_suffix}"

    return base_name


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_runtime_config(args):
    runtime_config = {}
    for key, value in vars(args).items():
        runtime_config[key] = value
    return runtime_config


def save_runtime_config(args):
    config_dst_path = os.path.join(args.save, 'config.yaml')
    with open(config_dst_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            build_runtime_config(args),
            f,
            sort_keys=False,
            allow_unicode=False,
        )
    print("Saved experiment config to %s" % config_dst_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='DconnNet Training With Pytorch')

    parser.add_argument('--device', type=int, default=1,)
    parser.add_argument('--seed', type=int, default=42,
                        help='global random seed for reproducible training/evaluation')

    # dataset info
    parser.add_argument('--dataset', type=str, default='drive',
                        help='isic, chase, drive, octa500-3M, octa500-6M, cremi, deepglobe')

    parser.add_argument('--data_root', type=str, default='data/DRIVE',
                        help='dataset directory (e.g. data/CREMI for --dataset cremi)')
    parser.add_argument('--resize', type=int, default=[256, 256], nargs='+',
                        help='image size: [height, width]')
    parser.add_argument('--label_mode', type=str, default='binary',
                        choices=['binary', 'dist', 'dist_inverted'],
                        help='label mode: binary, dist, dist_inverted')

    # network option & hyper-parameters
    parser.add_argument('--model', type=str, default='dconnnet',
                        help='model name registered in model/registry.py (e.g. dconnnet, unet)')
    parser.add_argument('--num_channels', type=int, default=None,
                        help='입력 이미지 채널 수 (unet 기본 3, frunet 기본 1)')
    parser.add_argument('--diff_timesteps', type=int, default=100,
                        help='확산 모델 타임스텝 수')
    parser.add_argument('--diff_objective', type=str, default='pred_noise',
                        choices=['pred_noise', 'pred_x0', 'pred_v'],
                        help='확산 모델 학습 목표')
    parser.add_argument('--beta_schedule', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='확산 스케줄러 타입')
    parser.add_argument('--image_size', type=int, default=128,
                        help='MedSegDiff 입력 이미지 크기')
    parser.add_argument('--num-class', type=int, default=4, metavar='N',
                        help='number of classes for your data')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='DataLoader worker processes (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--max_train_batches', type=int, default=0,
                        help='limit train batches per epoch (<=0 disables limit)')
    parser.add_argument('--max_eval_batches', type=int, default=0,
                        help='limit validation/test batches per evaluation pass (<=0 disables limit)')
    parser.add_argument('--lr', type=float, default=0.00085, metavar='LR',
                        help='learning rate (default: 8.5e-4)')
    parser.add_argument('--lr-update', type=str, default='step',
                        help='the lr update strategy: poly, step, warm-up-epoch, CosineAnnealingWarmRestarts, CosineAnnealingLR')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                        help='optimizer: adam (default) or adamw')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay for optimizer (default: 0.0)')
    parser.add_argument('--lr-step', type=int, default=12,
                        help='define only when you select step lr optimization: what is the step size for reducing your lr')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='define only when you select step lr optimization: what is the annealing rate for reducing your lr (lr = lr*gamma)')

    parser.add_argument('--folds', type=int, default=1,
                        help='define folds number K for K-fold validation')
    parser.add_argument('--target_fold', type=int, default=None,
                        help='1-based fold index to run; default runs all folds')
    parser.add_argument('--conn_num', type=int, default=8, choices=[8, 24],
                        help='the number of connections for DconnNet (supported: 8, 24)')
    parser.add_argument('--conn_layout', type=str, default=None,
                        choices=['standard8', 'full24', 'mid8', 'out8', 'scale3_8', 'scale4_8'],
                        help='connectivity layout: default is standard8 for conn_num=8 and full24 for conn_num=24')
    parser.add_argument('--final_decision', type=str, default='max',
                        choices=['max', 'mean', 'weighted_mean'],
                        help='final bilateral voting reduction (non-max requires standard8)')
    parser.add_argument('--conn_fusion', type=str, default='none',
                        choices=['none', 'gate', 'scaled_sum', 'conv_residual', 'decoder_guided', 'dg', 'dg_direct', 'dsf', 'dg_dsf'],
                        help='fork-specific directional fusion mode; none keeps legacy path')
    parser.add_argument('--fusion_branches', type=str, default='io',
                        help='active connectivity branches: i=inner(3x3), m=mid8, o=out8, p=scale3_8(7x7), f=scale4_8(9x9)')
    parser.add_argument('--fusion_lambda_middle', type=float, default=0.05,
                        help='lambda for middle branch affinity term')
    parser.add_argument('--fusion_lambda_fused', type=float, default=0.3,
                        help='lambda for fused branch affinity term')
    parser.add_argument('--fusion_residual_scale', type=float, default=0.2,
                        help='residual scale for conn_fusion=scaled_sum')

    # New options for SegAux and DGRF
    parser.add_argument('--use_seg_aux', action='store_true', default=False,
                        help='Enable Segmentation Auxiliary Supervision')
    parser.add_argument('--seg_aux_weight', type=float, default=None,
                        help='weight for SegAux BCE loss')
    parser.add_argument('--fusion_gate_reg_weight', type=float, default=0.01,
                        help='weight for DGRF gate regularization loss')
    parser.add_argument('--conn_aux_c3_weight', type=float, default=0.3,
                        help='weight for auxiliary C3 (inner) loss when DGRF is enabled')
    parser.add_argument('--conn_aux_c4_weight', type=float, default=0.2,
                        help='weight for auxiliary C4 (middle) loss when DGRF is enabled')
    parser.add_argument('--conn_aux_c5_weight', type=float, default=0.2,
                        help='weight for auxiliary C5 (outer) loss when DGRF is enabled')
    parser.add_argument('--conn_aux_c6_weight', type=float, default=0.3,
                        help='weight for auxiliary C6 (scale3_8) loss when DGRF is enabled')
    parser.add_argument('--conn_aux_c7_weight', type=float, default=0.2,
                        help='weight for auxiliary C7 (scale4_8) loss when DGRF is enabled')
    parser.add_argument('--legacy_aux_weight', type=float, default=0.0,
                        help=('fork-specific: when conn_fusion is enabled, add legacy_aux_weight * loss(mapped_c5) '
                              'to the fusion loss; default 0.0 keeps current fusion behavior. '
                              'Baseline non-fusion path is unaffected (it always uses 0.3 internally).'))
    parser.add_argument('--debug_loss_breakdown', action='store_true', default=False,
                        help=('fork-specific: write per-epoch averaged loss term values to '
                              '<save>/loss_breakdown.csv for diagnostic analysis. Default off.'))
    parser.add_argument('--simple_loss', action='store_true', default=False,
                        help='Use BCE+Dice loss only without connectivity supervision (for baseline comparison models)')

    parser.add_argument('--tau', type=float, default=3.0,
                        help='the temperature parameter tau for the distance connectivity loss')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='the weight parameter sigma for the distance connectivity loss')
    parser.add_argument('--dist_aux_loss', type=str, default='smooth_l1',
                        choices=['smooth_l1', 'gjml_sf_l1', 'cl_dice'],
                        help='auxiliary regression loss for dist/dist_inverted affinity targets')
    parser.add_argument('--dist_sf_l1_gamma', type=float, default=1.0,
                        help='gamma for Stable Focal-L1 when --dist_aux_loss=gjml_sf_l1')

    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save', default='save',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--output_dir', type=str, default='output/',
                        help='base output directory; save path becomes output_dir/<fold_scope>/<experiment_name> (binary: binary_conn_bce, dist: label_mode_conn_dist_aux_loss)')
    parser.add_argument('--experiment_name_suffix', type=str, default='',
                        help='optional suffix appended to experiment output folder name')

    parser.add_argument('--save-per-epochs', type=int, default=50,
                        help='per epochs to save')
    parser.add_argument('--monitor_metric', '--monitor-metric', type=str, default='val_dice',
                        choices=['val_dice', 'val_loss'],
                        help='metric for best checkpoint and early stopping')
    parser.add_argument('--early_stopping_patience', '--early-stopping-patience', type=int, default=20,
                        help='patience for validation-based early stopping (<=0 disables early stopping)')
    parser.add_argument('--early_stopping_min_delta', '--early-stopping-min-delta', type=float, default=0.001,
                        help='minimum improvement threshold for monitor metric')
    parser.add_argument('--early_stopping_tie_eps', '--early-stopping-tie-eps', type=float, default=1e-4,
                        help='tie epsilon for val_dice tie-break using val_loss')
    parser.add_argument('--early_stopping_stop_interval', '--early-stopping-stop-interval', type=int, default=10,
                        help='actual stopping happens only on epochs divisible by this interval')
    parser.add_argument('--tie_break_with_loss', '--tie-break-with-loss', dest='tie_break_with_loss',
                        action='store_true', default=True,
                        help='when val_dice is tied within eps, prefer lower val_loss')
    parser.add_argument('--no_tie_break_with_loss', '--no-tie-break-with-loss', dest='tie_break_with_loss',
                        action='store_false',
                        help='disable val_loss tie-break when val_dice values are tied')
    parser.add_argument('--save_best_only', '--save-best-only', dest='save_best_only',
                        action='store_true', default=False,
                        help='if set, disable periodic epoch checkpoints and keep best checkpoint only')
    parser.add_argument('--no_save_best_only', '--no-save-best-only', dest='save_best_only',
                        action='store_false',
                        help='enable periodic epoch checkpoints')

    # evaluation only
    parser.add_argument('--external_data', type=str, default=None,
                        help='external validation dataset name; output is saved to output_dir/drive_to_{name}/ instead of output_dir/{dataset}/')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='test only, please load the pretrained model')
    args = parser.parse_args()

    if args.dataset == 'deepglobe':
        if args.data_root == 'data/DRIVE':
            args.data_root = 'data/DeepGlobe/splitted'
        if list(args.resize) == [256, 256]:
            args.resize = [1024, 1024]

    try:
        args.conn_layout = normalize_conn_layout(args.conn_num, args.conn_layout)
    except ValueError as exc:
        parser.error(str(exc))
    args.conn_fusion = normalize_conn_fusion_mode(args.conn_fusion)
    try:
        args.final_decision = normalize_final_decision(args.final_decision)
    except ValueError as exc:
        parser.error(str(exc))

    if args.output_dir:
        if getattr(args, 'model', 'dconnnet') != 'dconnnet':
            args.save = os.path.join(
                args.output_dir,
                'baseline',
                args.dataset,
                args.model,
            )
        else:
            output_scope = f"drive_to_{args.external_data}" if args.external_data else args.dataset
            args.save = os.path.join(
                args.output_dir,
                output_scope,
                get_experiment_output_name(args),
            )

    os.makedirs(args.save, exist_ok=True)

    if args.target_fold is not None:
        if args.target_fold < 1 or args.target_fold > args.folds:
            parser.error('--target_fold must be within [1, --folds]')

    if str(args.dataset).startswith('retouch'):
        parser.error('RETOUCH dataset is no longer supported in active training workflow')

    if args.use_seg_aux and args.seg_aux_weight is None:
        parser.error('--use_seg_aux requires explicit --seg_aux_weight')

    if args.num_class != 1 and args.conn_layout != 'standard8':
        parser.error('multi-class mode currently supports only conn_layout=standard8')
    if args.conn_fusion != 'none':
        if args.num_class != 1:
            parser.error('conn_fusion currently supports only num_class=1')
        if args.conn_num != 8:
            parser.error('conn_fusion currently supports only conn_num=8')
        if args.conn_layout != 'standard8':
            parser.error('conn_fusion currently supports only conn_layout=standard8')

    if args.final_decision != 'max' and args.conn_layout != 'standard8':
        parser.error(
            'final_decision other than max currently supports only conn_layout=standard8'
        )
    args.connectivity_layout = resolve_connectivity_layout(args.conn_num, args.conn_layout)
    # Keep the CLI-level `conn_num` intact for layout resolution semantics and
    # expose directional-channel count explicitly for downstream consumers.
    args.conn_channels = args.connectivity_layout['channel_count']

    if args.early_stopping_patience < 0:
        parser.error('--early_stopping_patience must be >= 0')
    if args.early_stopping_min_delta < 0:
        parser.error('--early_stopping_min_delta must be >= 0')
    if args.early_stopping_tie_eps < 0:
        parser.error('--early_stopping_tie_eps must be >= 0')
    if args.early_stopping_stop_interval < 1:
        parser.error('--early_stopping_stop_interval must be >= 1')
    if args.max_train_batches < 0:
        parser.error('--max_train_batches must be >= 0')
    if args.max_eval_batches < 0:
        parser.error('--max_eval_batches must be >= 0')

    # Group related runtime controls for clearer downstream consumption.
    args.monitoring_config = {
        'monitor_metric': args.monitor_metric,
    }
    args.early_stopping_config = {
        'patience': int(args.early_stopping_patience),
        'min_delta': float(args.early_stopping_min_delta),
        'tie_eps': float(args.early_stopping_tie_eps),
        'stop_interval': int(args.early_stopping_stop_interval),
        'tie_break_with_loss': bool(args.tie_break_with_loss),
    }
    args.checkpoint_config = {
        'save_best_only': bool(args.save_best_only),
        'save_per_epochs': int(args.save_per_epochs),
    }

    return args


def main(args):

    torch.cuda.set_device(args.device)  # GPU id
    seed_everything(args.seed)

    print("Output directory: %s" % args.save)
    save_runtime_config(args)
    create_exp_directory(args.save)

    seed_everything(args.seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    trainset, validset, testset = build_datasets(args)
    train_loader, val_loader, test_loader = build_dataloaders(
        args=args,
        trainset=trainset,
        validset=validset,
        testset=testset,
        seed_worker=seed_worker,
        loader_generator=loader_generator,
    )

    batch_info = [f"batch number: {len(train_loader)}"]
    if validset is not None:
        batch_info.append(f"Validation batch number: {len(val_loader)}")
    if testset is not None:
        batch_info.append(f"Test batch number: {len(test_loader)}")
    print(" | ".join(batch_info))

    model = _build_model(args.model, args).cuda()

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=torch.device('cpu')))
        model = model.cuda()
        print("Loaded pretrained model from %s" % args.pretrained)

    solver = Solver(args)

    solver.train(model, train_loader, val_loader,
                 num_epochs=args.epochs, label_mode=args.label_mode,
                 test_loader=test_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
