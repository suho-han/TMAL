import math
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from connect_loss import (
    Bilateral_voting,
    Bilateral_voting_kxk,
    normalize_final_decision,
    resolve_connectivity_layout,
)
from lr_update import get_lr
from metrics.cal_betti import getBettiErrors
from metrics.cldice import clDice
from model.DconnNet import normalize_conn_fusion_mode
from src.engine import run_test_only_eval, run_validation_epoch
from src.losses import build_loss_functions
from src.metrics import compute_binary_precision_accuracy
from src.metrics import per_class_dice as compute_per_class_dice
from src.metrics.hd95 import hd95
from src.utils import (EarlyStopping, LossBreakdownLogger, build_epoch_postfix, build_progress_prefix, build_sample_metric_row, create_exp_directory, create_safe_summary_writer, format_elapsed_hms,
                       is_nan_metric, load_best_checkpoint, load_resume_checkpoint, log_epoch_scalars, resolve_training_runtime_configs, save_resume_checkpoint, write_epoch_result_row,
                       write_eval_summary, write_result_txt)
from src.utils.batch_utils import unpack_test_batch
from src.utils.results import write_sample_metrics_csv
from src.utils.visualization import save_checkpoint_batch_triplet as _save_batch_triplet

try:
    from apex import amp
except Exception:
    class _AmpFallback:
        def initialize(self, model, optim, opt_level='O2'):
            return model, optim

        @contextmanager
        def scale_loss(self, loss, optimizer):
            yield loss

    amp = _AmpFallback()


def _nanmean(lst):
    valid = [v for v in lst if not math.isnan(float(v))]
    return float(np.mean(valid)) if valid else float('nan')


def _nanstd(lst):
    valid = [v for v in lst if not math.isnan(float(v))]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def _fmt_ms(m, s, fmt='7.4f'):
    if math.isnan(m):
        return '      nan'
    return f'{m:{fmt}}±{s:.4f}'


def _blank_epoch_metrics(train_loss=float('nan')):
    return {
        'dice': float('nan'),
        'jac': float('nan'),
        'cldice': float('nan'),
        'precision': float('nan'),
        'accuracy': float('nan'),
        'betti_error_0': float('nan'),
        'betti_error_1': float('nan'),
        'hd95': float('nan'),
        'val_loss': float('nan'),
        'val_loss_c5': float('nan'),
        'train_loss': train_loss,
        'val_loss_terms': {},
    }


class Solver(object):
    def __init__(self, args, optim=torch.optim.Adam):
        self.args = args
        self.optim = optim
        self.NumClass = self.args.num_class
        self.lr = self.args.lr
        self.conn_fusion = normalize_conn_fusion_mode(getattr(self.args, 'conn_fusion', 'none'))
        self.fusion_enabled = self.conn_fusion != 'none'
        self.fusion_branches = str(getattr(self.args, 'fusion_branches', 'io'))
        self.fusion_lambda_middle = float(getattr(self.args, 'fusion_lambda_middle', 0.05))
        self.fusion_lambda_fused = float(getattr(self.args, 'fusion_lambda_fused', 0.3))
        self.connectivity_layout = resolve_connectivity_layout(
            self.args.conn_num,
            getattr(self.args, 'conn_layout', None),
        )
        self.final_decision = normalize_final_decision(
            getattr(self.args, 'final_decision', 'max')
        )
        if (
            self.final_decision != 'max'
            and self.connectivity_layout['name'] != 'standard8'
        ):
            raise ValueError(
                "final_decision other than 'max' is supported only for "
                "conn_layout='standard8'"
            )
        self.conn_channels = self.connectivity_layout['channel_count']
        if self.fusion_enabled:
            if self.NumClass != 1:
                raise ValueError("conn_fusion currently supports only num_class=1")
            if self.conn_channels != 8:
                raise ValueError("conn_fusion currently supports only conn_num=8")
            if self.connectivity_layout['name'] != 'standard8':
                raise ValueError("conn_fusion currently supports only conn_layout=standard8")
        H, W = args.resize

        self.hori_translation = torch.zeros([1, self.NumClass, W, W])
        for i in range(W - 1):
            self.hori_translation[:, :, i, i + 1] = torch.tensor(1.0)

        self.verti_translation = torch.zeros([1, self.NumClass, H, H])
        for j in range(H - 1):
            self.verti_translation[:, :, j, j + 1] = torch.tensor(1.0)

        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()

    def dist_to_binary(self, x, label_mode):
        """
        Convert a distance-label ground truth tensor to a binary mask.

        Expected input shapes:
            - (B, C, 8, H, W)
            - or any tensor where foreground/background thresholding is elementwise.

        IMPORTANT:
            If your signed distance map defines foreground as negative values,
            change the 'dist' branch from (x > 0) to (x < 0).
        """
        if label_mode in ['dist', 'dist_inverted']:
            return (x > 0).float()

        else:
            return (x > 0.5).float()

    def dist_score_to_binary(self, x):
        """
        Convert the distance-mode prediction score map to a binary mask.

        In the dist path, `pred_score` is produced from sigmoid affinities via
        bilateral voting, so it is non-negative almost everywhere. Thresholding
        at `> 0` collapses to all-foreground masks; use a probability-style
        threshold instead.
        """
        return (x > 0.5).float()

    def apply_connectivity_voting(self, conn_map, hori_translation, verti_translation):
        """
        Convert directional affinity / connectivity maps to a voted score map.
        """
        layout = self.connectivity_layout
        if layout['name'] == 'standard8':
            pred_map, vote_map = Bilateral_voting(
                conn_map,
                hori_translation,
                verti_translation,
                final_decision=self.final_decision,
            )
        else:
            pred_map, vote_map = Bilateral_voting_kxk(
                conn_map,
                hori_translation,
                verti_translation,
                conn_num=layout['kernel_size'],
                offsets=layout['offsets'],
            )
        return pred_map, vote_map

    def connectivity_to_mask(self, conn_map, hori_translation, verti_translation):
        """
        Convert connectivity map to final binary segmentation mask
        using bilateral voting.
        """
        pred_mask, _ = self.apply_connectivity_voting(
            conn_map, hori_translation, verti_translation
        )
        pred_mask = (pred_mask > 0).float()
        return pred_mask

    def _unpack_model_outputs(self, model_output):
        if isinstance(model_output, dict):
            if self.fusion_enabled:
                required = {'fused', 'inner'}
                missing = sorted(required - set(model_output.keys()))
                if missing:
                    raise ValueError(f"Fusion model output is missing keys: {missing}")
            else:
                required = {'fused', 'aux'}
                missing = sorted(required - set(model_output.keys()))
                if missing:
                    raise ValueError(f"Baseline model output is missing keys: {missing}")

            return model_output

        if isinstance(model_output, (tuple, list)) and len(model_output) == 2:
            return {
                'fused': model_output[0],
                'aux': model_output[1],
            }

        raise TypeError(
            "Unexpected model output type. Expected tuple(len=2) for legacy mode "
            "or dict with keys {fused, inner, outer} for fusion mode."
        )

    def _compute_fusion_profile_loss(self, output_dict, target, binary_gt=None, collect_edge_stats=False):
        if not self.fusion_enabled:
            raise RuntimeError("_compute_fusion_profile_loss can be used only when conn_fusion is enabled")

        fused_logits = output_dict['fused']
        inner_logits = output_dict['inner']

        if (
            collect_edge_stats
            and self.loss_func.label_mode in ['dist', 'dist_inverted']
            and hasattr(self.loss_func, 'set_dist_edge_stat_collection')
        ):
            self.loss_func.set_dist_edge_stat_collection(True)
        _, fused_terms = self.loss_func(fused_logits, target, return_details=True)
        if (
            collect_edge_stats
            and self.loss_func.label_mode in ['dist', 'dist_inverted']
            and hasattr(self.loss_func, 'set_dist_edge_stat_collection')
        ):
            self.loss_func.set_dist_edge_stat_collection(False)

        _, inner_terms = self.loss_func(inner_logits, target, return_details=True)

        middle_terms = None
        if 'middle' in output_dict and self.loss_func_middle is not None:
            _, middle_terms = self.loss_func_middle(output_dict['middle'], target, return_details=True)

        outer_terms = None
        if 'outer' in output_dict and self.loss_func_outer is not None:
            _, outer_terms = self.loss_func_outer(output_dict['outer'], target, return_details=True)
        scale3_terms = None
        if 'scale3' in output_dict and self.loss_func_scale3 is not None:
            _, scale3_terms = self.loss_func_scale3(output_dict['scale3'], target, return_details=True)
        scale4_terms = None
        if 'scale4' in output_dict and self.loss_func_scale4 is not None:
            _, scale4_terms = self.loss_func_scale4(output_dict['scale4'], target, return_details=True)

        fused_affinity = fused_terms['affinity']
        seg = fused_terms['vote'] + fused_terms['dice']
        total = seg + self.fusion_lambda_fused * fused_affinity
        terms = {
            'total': total,
            'seg': seg,
            'vote': fused_terms['vote'],
            'dice': fused_terms['dice'],
            'fused_affinity': fused_affinity,
        }

        active_conn_fusion = getattr(
            self,
            'conn_fusion',
            normalize_conn_fusion_mode(getattr(self.args, 'conn_fusion', 'none')),
        )
        if active_conn_fusion in {'dg', 'dg_direct', 'dsf', 'dg_dsf'}:
            base_fused_total = total
            c3_aux = inner_terms.get('total', inner_terms['affinity'])
            c3_weight = float(getattr(self.args, 'conn_aux_c3_weight', 0.0))
            total = base_fused_total + c3_weight * c3_aux
            terms['dgrf_fused_main'] = base_fused_total
            terms['dgrf_c3_aux'] = c3_aux

            if middle_terms is not None:
                c4_aux = middle_terms.get('total', middle_terms['affinity'])
                c4_weight = float(getattr(self.args, 'conn_aux_c4_weight', 0.0))
                total = total + c4_weight * c4_aux
                terms['dgrf_c4_aux'] = c4_aux

            if outer_terms is not None:
                c5_aux = outer_terms.get('total', outer_terms['affinity'])
                c5_weight = float(getattr(self.args, 'conn_aux_c5_weight', 0.0))
                total = total + c5_weight * c5_aux
                terms['dgrf_c5_aux'] = c5_aux

            if scale3_terms is not None:
                c6_aux = scale3_terms.get('total', scale3_terms['affinity'])
                c6_weight = float(getattr(self.args, 'conn_aux_c6_weight', 0.0))
                total = total + c6_weight * c6_aux
                terms['dgrf_c6_aux'] = c6_aux

            if scale4_terms is not None:
                c7_aux = scale4_terms.get('total', scale4_terms['affinity'])
                c7_weight = float(getattr(self.args, 'conn_aux_c7_weight', 0.0))
                total = total + c7_weight * c7_aux
                terms['dgrf_c7_aux'] = c7_aux

            if 'fusion_gate' in output_dict:
                gate_loss = output_dict['fusion_gate'].mean()
                total = total + float(getattr(self.args, 'fusion_gate_reg_weight', 0.0)) * gate_loss
                terms['gate'] = gate_loss

        total, terms = self._apply_seg_aux(output_dict, binary_gt, total, terms)

        legacy_aux_weight = float(getattr(self.args, 'legacy_aux_weight', 0.0))
        debug_breakdown = bool(getattr(self.args, 'debug_loss_breakdown', False))
        if (legacy_aux_weight > 0.0 or debug_breakdown) and 'aux' in output_dict:
            _, aux_terms = self.loss_func(output_dict['aux'], target, return_details=True)
            aux_total = aux_terms.get('total', aux_terms.get('affinity'))
            if legacy_aux_weight > 0.0:
                total = total + legacy_aux_weight * aux_total
                terms['legacy_aux'] = aux_total
            elif debug_breakdown:
                terms['legacy_aux_monitor'] = aux_total

        terms['total'] = total

        return total, terms

    def _apply_seg_aux(self, model_output, binary_gt, loss, loss_dict):
        """Apply seg_aux loss if enabled and mask_logit is present. Returns (updated_loss, updated_dict)."""
        if not (getattr(self.args, 'use_seg_aux', False) and 'mask_logit' in model_output and binary_gt is not None):
            return loss, loss_dict
        gt = binary_gt.unsqueeze(1) if binary_gt.ndim == 3 else binary_gt
        seg_loss = F.binary_cross_entropy_with_logits(model_output['mask_logit'], gt)
        loss = loss + getattr(self.args, 'seg_aux_weight', 0.0) * seg_loss
        loss_dict = {**loss_dict, 'seg_aux': seg_loss}
        return loss, loss_dict

    def _train_step(self, sample_batched, net, optimizer, global_step, writer, breakdown_logger, label_mode):
        """Process one training batch. Returns (loss_value, new_global_step)."""
        if isinstance(sample_batched, dict):
            X = sample_batched['image']
            y = sample_batched['label']
            binary_gt = sample_batched.get('binary_gt', None)
        else:
            X = sample_batched[0]
            y = sample_batched[1]
            binary_gt = None

        X = X.cuda()
        y = y.float().cuda()
        if binary_gt is not None:
            binary_gt = binary_gt.cuda()

        optimizer.zero_grad()
        model_output = self._unpack_model_outputs(net(X))

        if self.fusion_enabled:
            loss, loss_main_dict = self._compute_fusion_profile_loss(
                model_output, y, binary_gt=binary_gt, collect_edge_stats=True,
            )
            loss_aux_dict = {}
        else:
            output = model_output['fused']
            aux_out = model_output['aux']
            if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'set_dist_edge_stat_collection'):
                self.loss_func.set_dist_edge_stat_collection(True)
            loss_main, loss_main_dict = self.loss_func(output, y, return_details=True)
            if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'set_dist_edge_stat_collection'):
                self.loss_func.set_dist_edge_stat_collection(False)
            loss_aux, loss_aux_dict = self.loss_func(aux_out, y, return_details=True)
            loss = loss_main + 0.3 * loss_aux
            loss, loss_main_dict = self._apply_seg_aux(model_output, binary_gt, loss, loss_main_dict)

        with amp.scale_loss(loss, optimizer) as scaled:
            scaled.backward()
        optimizer.step()

        writer.add_scalar('train/loss_total', float(loss.item()), global_step)
        for key, value in loss_main_dict.items():
            writer.add_scalar(f'train/main/{key}', float(value.detach().item()), global_step)
        for key, value in loss_aux_dict.items():
            writer.add_scalar(f'train/aux/{key}', float(value.detach().item()), global_step)
        if breakdown_logger is not None:
            batch_terms = {'loss_total': float(loss.detach().item())}
            for key, value in loss_main_dict.items():
                batch_terms[key] = float(value.detach().item())
            for key, value in loss_aux_dict.items():
                batch_terms[f'aux_{key}'] = float(value.detach().item())
            breakdown_logger.accumulate(batch_terms)

        return float(loss.item()), global_step + 1

    def _resolve_training_runtime_configs(self):
        return resolve_training_runtime_configs(self.args)

    def train(self, model, train_loader, val_loader, num_epochs=10, label_mode='binary', test_loader=None):
        optimizer_name = str(getattr(self.args, 'optimizer', 'adam')).lower()
        weight_decay = float(getattr(self.args, 'weight_decay', 0.0))
        if optimizer_name == 'adamw':
            optim = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            optim = self.optim(model.parameters(), lr=self.lr, weight_decay=weight_decay)
        max_train_batches = int(getattr(self.args, 'max_train_batches', 0))
        if max_train_batches <= 0:
            max_train_batches = None

        print('START TRAIN.')
        create_exp_directory(self.args.save)
        tb_dir = os.path.join(self.args.save, 'tensorboard', f'exp')
        writer = create_safe_summary_writer(SummaryWriter, tb_dir)

        (
            self.loss_func,
            self.loss_func_middle,
            self.loss_func_outer,
            self.loss_func_scale3,
            self.loss_func_scale4,
        ) = build_loss_functions(
            args=self.args,
            hori_translation=self.hori_translation,
            verti_translation=self.verti_translation,
            label_mode=label_mode,
            fusion_enabled=self.fusion_enabled,
            fusion_branches=self.fusion_branches,
        )
        net, optimizer = amp.initialize(model, optim, opt_level='O2')

        start_epoch, _resume_global_step, _resume_ckpt = load_resume_checkpoint(self.args.save)
        if _resume_ckpt is not None:
            net.load_state_dict(_resume_ckpt['model_state_dict'])
            try:
                optimizer.load_state_dict(_resume_ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f'[WARN] Could not restore optimizer state: {e}')
            print(f'RESUME from epoch {start_epoch}, global_step {_resume_global_step}.')

        monitor_metric, save_best_only, save_per_epochs, early_stopping_kwargs = (
            self._resolve_training_runtime_configs()
        )
        monitor_mode = 'max' if monitor_metric == 'val_dice' else 'min'
        early_stopper = None
        if val_loader is not None:
            early_stopper = EarlyStopping(
                monitor_metric=monitor_metric,
                mode=monitor_mode,
                **early_stopping_kwargs,
            )

        # Use raw optim (not the amp-wrapped optimizer) so scheduler works for both paths
        scheduled_modes = ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR']
        if self.args.lr_update in scheduled_modes:
            scheduled = True
            if self.args.lr_update == 'CosineAnnealingWarmRestarts':
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=15, T_mult=2, eta_min=0.00001)
            elif self.args.lr_update == 'CosineAnnealingLR':
                scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=1e-6)
        else:
            scheduled = False

        breakdown_logger = (
            LossBreakdownLogger(self.args.save)
            if bool(getattr(self.args, 'debug_loss_breakdown', False))
            else None
        )

        if self.args.test_only:
            run_test_only_eval(self, net, val_loader, test_loader, writer)
        else:
            global_step = _resume_global_step
            train_start_time = time.perf_counter()
            completed_epoch = start_epoch
            progress_prefix = build_progress_prefix(self.args)
            epoch_progress = tqdm(
                range(start_epoch, self.args.epochs),
                desc=f'{progress_prefix} Epoch',
                dynamic_ncols=True,
                leave=True,
            )
            epoch_metrics = _blank_epoch_metrics()
            for epoch in epoch_progress:
                current_epoch = epoch + 1
                completed_epoch = current_epoch
                net.train()
                train_loss_epoch = []
                epoch_progress.set_description(f'{progress_prefix} Epoch {current_epoch}/{self.args.epochs}')
                if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'reset_dist_edge_stats'):
                    self.loss_func.reset_dist_edge_stats()

                if scheduled:
                    scheduler.step()
                else:
                    curr_lr = get_lr(
                        self.lr,
                        self.args.lr_update,
                        epoch,
                        num_epochs,
                        gamma=self.args.gamma,
                        step=self.args.lr_step
                    )
                    for param_group in optim.param_groups:
                        param_group['lr'] = curr_lr

                with tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f'{progress_prefix} Train',
                    dynamic_ncols=True,
                    leave=False,
                ) as train_progress:
                    for i_batch, sample_batched in train_progress:
                        loss_val, global_step = self._train_step(
                            sample_batched, net, optimizer, global_step, writer, breakdown_logger, label_mode,
                        )
                        train_loss_epoch.append(loss_val)
                        train_progress.set_postfix(
                            loss=f'{loss_val:.4f}',
                            lr=f'{float(optim.param_groups[0]["lr"]):.2e}',
                        )
                        if max_train_batches is not None and (i_batch + 1) >= max_train_batches:
                            break

                mean_train_loss = float(np.mean(train_loss_epoch)) if train_loss_epoch else float('nan')
                should_save_batch_triplet = ((epoch + 1) % save_per_epochs == 0)

                epoch_metrics = _blank_epoch_metrics(mean_train_loss)

                stop_training = False
                if val_loader is not None:
                    epoch_metrics, stop_training = run_validation_epoch(
                        solver=self,
                        net=net,
                        model=model,
                        optimizer=optimizer,
                        val_loader=val_loader,
                        epoch=epoch,
                        current_epoch=current_epoch,
                        mean_train_loss=mean_train_loss,
                        save_batch_triplet=should_save_batch_triplet,
                        monitor_metric=monitor_metric,
                        early_stopper=early_stopper,
                    )

                if (not save_best_only) and (current_epoch % save_per_epochs == 0):
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.args.save, 'models', str(current_epoch) + '_model.pth')
                    )
                    save_resume_checkpoint(
                        self.args.save, model.state_dict(), optimizer.state_dict(),
                        current_epoch, global_step,
                    )

                dist_edge_stats = (
                    self.loss_func.get_dist_edge_stats()
                    if label_mode in ['dist', 'dist_inverted'] and hasattr(self.loss_func, 'get_dist_edge_stats')
                    else None
                )
                log_epoch_scalars(
                    writer=writer,
                    current_epoch=current_epoch,
                    mean_train_loss=mean_train_loss,
                    epoch_metrics=epoch_metrics,
                    early_stopper=early_stopper,
                    val_loader=val_loader,
                    dist_edge_stats=dist_edge_stats,
                )

                epoch_postfix = build_epoch_postfix(
                    mean_train_loss=mean_train_loss,
                    val_loader=val_loader,
                    epoch_metrics=epoch_metrics,
                    early_stopper=early_stopper,
                    dist_edge_stats=dist_edge_stats,
                )
                epoch_progress.set_postfix(epoch_postfix)
                elapsed_hms = format_elapsed_hms(time.perf_counter() - train_start_time)
                write_epoch_result_row(self.args.save, current_epoch, epoch_metrics, elapsed_hms)
                if breakdown_logger is not None:
                    breakdown_logger.flush(current_epoch)
                if stop_training:
                    break
            epoch_progress.close()

            if completed_epoch == 0:
                raise ValueError('Training loop completed zero epochs; please set --epochs >= 1.')

            final_eval_split = 'test' if test_loader is not None else 'val'
            final_eval_loader = test_loader if test_loader is not None else val_loader
            if final_eval_loader is None:
                raise ValueError('Final evaluation requires test_loader or val_loader, but both are None.')
            net, final_checkpoint_name = load_best_checkpoint(
                self.args.save, net, val_loader, final_eval_split
            )

            print(f'RUN FINAL {final_eval_split.upper()} EVAL AFTER TRAINING.')
            final_eval_epoch = int(completed_epoch)
            total_elapsed_hms = format_elapsed_hms(time.perf_counter() - train_start_time)
            final_eval_metrics = self.test_epoch(
                net,
                final_eval_loader,
                final_eval_epoch - 1,
                train_loss=epoch_metrics['train_loss'],
                save_batch_triplet=True,
                split_name=final_eval_split,
            )
            write_eval_summary(
                save_dir=self.args.save,
                split_name='final',
                metrics=final_eval_metrics,
                checkpoint_name=final_checkpoint_name,
                evaluated_split=final_eval_split,
                eval_epoch=final_eval_epoch,
                elapsed_hms=total_elapsed_hms,
            )
            write_result_txt(
                save_dir=self.args.save,
                dataset=self.args.dataset,
                split_name='final',
                metrics=final_eval_metrics,
                checkpoint_name=final_checkpoint_name,
                evaluated_split=final_eval_split,
                eval_epoch=final_eval_epoch,
                elapsed_hms=total_elapsed_hms,
            )

            print('FINISH.')
            writer.close()

    def test_epoch(self, model, loader, epoch, train_loss=float('nan'), save_batch_triplet=False, split_name='test'):
        model.eval()
        max_eval_batches = int(getattr(self.args, 'max_eval_batches', 0)) or None

        dice_ls, jac_ls, cldc_ls, precision_ls, accuracy_ls = [], [], [], [], []
        betti_error_0_ls, betti_error_1_ls, hd95_ls, val_loss_ls = [], [], [], []
        sample_metric_rows = []
        val_term_sums, val_term_counts = {}, 0

        sample_csv = os.path.join(self.args.save, 'models', f'{split_name}_sample_metrics.csv')
        os.makedirs(os.path.dirname(sample_csv), exist_ok=True)
        progress_prefix = build_progress_prefix(self.args)

        with torch.no_grad():
            with tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f'{progress_prefix} {split_name.capitalize()} Eval',
                dynamic_ncols=True,
                leave=False,
            ) as eval_progress:
                for j_batch, test_data in eval_progress:
                    X_test, y_test_raw, sample_names, binary_gt = unpack_test_batch(test_data, j_batch)
                    X_test = X_test.cuda()
                    y_test_raw = y_test_raw.float().cuda()

                    model_output = self._unpack_model_outputs(model(X_test))
                    if self.fusion_enabled:
                        output_test = model_output['fused']
                        val_loss, val_main_dict = self._compute_fusion_profile_loss(
                            model_output, y_test_raw, binary_gt=binary_gt, collect_edge_stats=False,
                        )
                    else:
                        output_test = model_output['fused']
                        aux_out = model_output['aux']
                        val_loss_main, val_main_dict = self.loss_func(output_test, y_test_raw, return_details=True)
                        val_loss_aux, _ = self.loss_func(aux_out, y_test_raw, return_details=True)
                        val_loss = val_loss_main + 0.3 * val_loss_aux
                        val_loss, val_main_dict = self._apply_seg_aux(model_output, binary_gt, val_loss, val_main_dict)

                    if not val_loss.isnan():
                        val_loss_ls.append(val_loss.item())
                    if j_batch == 0:
                        val_term_sums = {k: 0.0 for k in val_main_dict}
                    for k, v in val_main_dict.items():
                        val_term_sums[k] += float(v.detach().item())
                    val_term_counts += 1

                    batch, _, H, W = X_test.shape
                    hori_translation = self.hori_translation.repeat(batch, 1, 1, 1).cuda()
                    verti_translation = self.verti_translation.repeat(batch, 1, 1, 1).cuda()

                    if self.args.num_class != 1:
                        raise NotImplementedError("Multi-class evaluation has been removed. Use --num-class 1.")

                    if self.loss_func.label_mode in ['dist', 'dist_inverted']:
                        pred_conn_prob = torch.sigmoid(output_test).view(
                            [batch, -1, self.conn_channels, H, W]
                        )
                        pred_score, _ = self.apply_connectivity_voting(
                            pred_conn_prob, hori_translation, verti_translation
                        )
                        pred = self.dist_score_to_binary(pred_score.view(batch, 1, H, W))
                        gt_mask = self.dist_to_binary(y_test_raw, self.loss_func.label_mode)
                    else:
                        output_prob = torch.sigmoid(output_test)
                        pred_conn_prob = output_prob.view([batch, -1, self.conn_channels, H, W])
                        pred_bin = (pred_conn_prob > 0.5).float()
                        gt_mask = (y_test_raw > 0.5).float()
                        pred = self.connectivity_to_mask(pred_bin, hori_translation, verti_translation)

                    if gt_mask.dim() == 3:
                        gt_mask = gt_mask.unsqueeze(1)

                    if save_batch_triplet:
                        _save_batch_triplet(self.args.save, X_test, pred, gt_mask, epoch + 1, j_batch, sample_names=sample_names)

                    dice, Jac = compute_per_class_dice(pred, gt_mask)
                    sample_cldc, sample_betti_error_0, sample_betti_error_1, sample_hd95 = [], [], [], []
                    if self.args.dataset in ('isic', 'cremi'):
                        sample_precision, sample_accuracy = compute_binary_precision_accuracy(pred, gt_mask)
                    else:
                        sample_precision = torch.full((batch,), float('nan'), device=pred.device)
                        sample_accuracy = torch.full((batch,), float('nan'), device=pred.device)
                    precision_ls += sample_precision.detach().cpu().tolist()
                    accuracy_ls += sample_accuracy.detach().cpu().tolist()
                    for b_idx in range(batch):
                        pred_np = pred[b_idx, 0].detach().cpu().numpy()
                        target_np = gt_mask[b_idx, 0].detach().cpu().numpy()
                        cldc = float(clDice(pred_np, target_np))
                        sample_cldc.append(cldc)
                        cldc_ls.append(cldc)
                        betti0_err, betti1_err = getBettiErrors(pred[b_idx, 0], gt_mask[b_idx, 0])
                        sample_betti_error_0.append(float(np.mean(betti0_err)) if len(betti0_err) > 0 else float('nan'))
                        sample_betti_error_1.append(float(np.mean(betti1_err)) if len(betti1_err) > 0 else float('nan'))
                        betti_error_0_ls.append(sample_betti_error_0[-1])
                        betti_error_1_ls.append(sample_betti_error_1[-1])
                        hd = float(hd95(pred_np, target_np))
                        sample_hd95.append(hd)
                        hd95_ls.append(hd)

                    sample_dice = dice[:, 0]
                    sample_jac = Jac[:, 0]
                    dice_ls += sample_dice.tolist()
                    jac_ls += sample_jac.tolist()

                    for b_idx in range(batch):
                        sample_metric_rows.append(build_sample_metric_row(
                            epoch=epoch + 1, batch_idx=j_batch, sample_in_batch=b_idx,
                            sample_name=sample_names[b_idx],
                            val_loss=val_loss.item(),
                            dice=sample_dice[b_idx].item(),
                            jac=sample_jac[b_idx].item(),
                            cldice=sample_cldc[b_idx],
                            precision=sample_precision[b_idx].item(),
                            accuracy=sample_accuracy[b_idx].item(),
                            betti_error_0=sample_betti_error_0[b_idx],
                            betti_error_1=sample_betti_error_1[b_idx],
                            hd95=sample_hd95[b_idx],
                        ))

                    if max_eval_batches is not None and (j_batch + 1) >= max_eval_batches:
                        break

        val_loss_terms = {k: v / val_term_counts for k, v in val_term_sums.items()} if val_term_counts > 0 else {}
        write_sample_metrics_csv(sample_csv, sample_metric_rows)

        res = {
            'dice':          _nanmean(dice_ls),
            'jac':           _nanmean(jac_ls),
            'cldice':        _nanmean(cldc_ls),
            'precision':     _nanmean(precision_ls),
            'accuracy':      _nanmean(accuracy_ls),
            'betti_error_0': _nanmean(betti_error_0_ls),
            'betti_error_1': _nanmean(betti_error_1_ls),
            'hd95':          _nanmean(hd95_ls),
            # STDs
            'dice_std':          _nanstd(dice_ls),
            'jac_std':           _nanstd(jac_ls),
            'cldice_std':        _nanstd(cldc_ls),
            'precision_std':     _nanstd(precision_ls),
            'accuracy_std':      _nanstd(accuracy_ls),
            'betti_error_0_std': _nanstd(betti_error_0_ls),
            'betti_error_1_std': _nanstd(betti_error_1_ls),
            'hd95_std':          _nanstd(hd95_ls),

            'val_loss': float(np.mean(val_loss_ls)) if val_loss_ls else float('nan'),
            'val_loss_c5' : float(val_loss_terms.get('dgrf_c5_aux', float('nan'))),
            'train_loss': float(train_loss),
            'val_loss_terms': val_loss_terms,
        }

        # Print per-sample results and summary
        tqdm.write('\n' + '-' * 80)
        tqdm.write(f"{'Sample':<25} {'Dice':>8} {'Jac':>8} {'clDice':>8} {'Betti0':>8} {'Betti1':>8} {'HD95':>8}")
        tqdm.write('-' * 80)
        for r in sample_metric_rows:
            def _f(v, fmt='8.4f'):
                return f'{v:{fmt}}' if not math.isnan(float(v)) else '     nan'
            tqdm.write(
                f"{str(r['sample_name']):<25} "
                f"{_f(r['dice'])} {_f(r['jac'])} {_f(r['cldice'])} "
                f"{_f(r['betti_error_0'], '8.2f')} {_f(r['betti_error_1'], '8.2f')} {_f(r['hd95'])}"
            )
        tqdm.write('-' * 80)


        tqdm.write(
            f"{'MEAN±STD':<25} "
            f"{_fmt_ms(res['dice'], res['dice_std'])} "
            f"{_fmt_ms(res['jac'], res['jac_std'])} "
            f"{_fmt_ms(res['cldice'], res['cldice_std'])} "
            f"{_fmt_ms(res['betti_error_0'], res['betti_error_0_std'], '7.2f')} "
            f"{_fmt_ms(res['betti_error_1'], res['betti_error_1_std'], '7.2f')} "
            f"{_fmt_ms(res['hd95'], res['hd95_std'])}"
        )
        tqdm.write('-' * 80 + '\n')

        return res
