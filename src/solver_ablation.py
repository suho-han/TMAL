"""Solver for baseline/ablation models — no connectivity supervision."""
import os
import time
import math

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from lr_update import get_lr
from metrics.cal_betti import getBettiErrors
from metrics.cldice import clDice
from src.metrics.hd95 import hd95
from src.engine import run_test_only_eval, run_validation_epoch
from src.losses import SimpleLoss
from src.metrics import compute_binary_precision_accuracy
from src.metrics import per_class_dice as compute_per_class_dice
from src.utils import (
    EarlyStopping,
    build_epoch_postfix,
    build_progress_prefix,
    build_sample_metric_row,
    create_exp_directory,
    create_safe_summary_writer,
    format_elapsed_hms,
    is_nan_metric,
    load_best_checkpoint,
    load_resume_checkpoint,
    log_epoch_scalars,
    resolve_training_runtime_configs,
    save_resume_checkpoint,
    write_epoch_result_row,
    write_eval_summary,
)
from src.utils.batch_utils import unpack_test_batch
from src.utils.results import save_best_checkpoint, write_sample_metrics_csv
from src.utils.visualization import save_checkpoint_batch_triplet

from solver import _nanmean, _nanstd, _fmt_ms

try:
    from apex import amp
except Exception:
    from solver import _AmpFallback
    amp = _AmpFallback()


class AblationSolver:
    """Solver for non-DconnNet comparison models.

    Supports:
    - UNet-style models with SimpleLoss (BCE + soft-Dice)
    - Diffusion models (segdiff, medsegdiff) via their own train_step/sample
    """

    def __init__(self, args, optim=torch.optim.Adam):
        self.args = args
        self.optim = optim
        self.NumClass = args.num_class
        self.lr = args.lr
        self._is_diffusion_model = False
        self.loss_func = None

    def _resolve_training_runtime_configs(self):
        return resolve_training_runtime_configs(self.args)

    def train(self, model, train_loader, val_loader, num_epochs=10, label_mode='binary', test_loader=None):
        optimizer_name = str(getattr(self.args, 'optimizer', 'adam')).lower()
        weight_decay = float(getattr(self.args, 'weight_decay', 0.0))
        if optimizer_name == 'adamw':
            optim = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            optim = self.optim(model.parameters(), lr=self.lr, weight_decay=weight_decay)

        max_train_batches = int(getattr(self.args, 'max_train_batches', 0)) or None

        print('START TRAIN.')
        create_exp_directory(self.args.save)
        tb_dir = os.path.join(self.args.save, 'tensorboard', 'exp')
        writer = create_safe_summary_writer(SummaryWriter, tb_dir)

        self._is_diffusion_model = hasattr(model, 'train_step')
        if self._is_diffusion_model:
            if hasattr(model, 'register_opt'):
                model.register_opt(optim)
            self.loss_func = None
            net, optimizer = model, optim
        else:
            self.loss_func = SimpleLoss(label_mode='binary')
            net, optimizer = amp.initialize(model, optim, opt_level='O2')

        start_epoch, _resume_global_step, _resume_ckpt = load_resume_checkpoint(self.args.save)
        if _resume_ckpt is not None:
            try:
                net.load_state_dict(_resume_ckpt['model_state_dict'])
            except Exception as e:
                print(f'[WARN] Could not restore model state: {e}')
            try:
                optimizer.load_state_dict(_resume_ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f'[WARN] Could not restore optimizer state: {e}')
            print(f'RESUME from epoch {start_epoch}, global_step {_resume_global_step}.')

        monitor_metric, save_best_only, save_per_epochs, early_stopping_kwargs = (
            self._resolve_training_runtime_configs()
        )
        monitor_mode = 'max' if monitor_metric == 'val_dice' else 'min'
        early_stopper = (
            EarlyStopping(monitor_metric=monitor_metric, mode=monitor_mode, **early_stopping_kwargs)
            if val_loader is not None else None
        )

        if self.args.lr_update in ('CosineAnnealingWarmRestarts', 'CosineAnnealingLR'):
            scheduled = True
            if self.args.lr_update == 'CosineAnnealingWarmRestarts':
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=15, T_mult=2, eta_min=0.00001)
            else:
                scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=1e-6)
        else:
            scheduled = False

        if self.args.test_only:
            run_test_only_eval(self, net, val_loader, test_loader, writer)
            return

        global_step = _resume_global_step
        train_start_time = time.perf_counter()
        completed_epoch = start_epoch
        progress_prefix = build_progress_prefix(self.args)

        _nan_epoch = {
            k: float('nan') for k in ['dice', 'jac', 'cldice', 'precision', 'accuracy',
                                        'betti_error_0', 'betti_error_1', 'val_loss', 'train_loss']
        }
        _nan_epoch['val_loss_terms'] = {}
        epoch_metrics = dict(_nan_epoch)

        epoch_progress = tqdm(
            range(start_epoch, self.args.epochs), desc=f'{progress_prefix} Epoch', dynamic_ncols=True, leave=True
        )

        for epoch in epoch_progress:
            current_epoch = epoch + 1
            completed_epoch = current_epoch
            net.train()
            train_loss_epoch = []
            epoch_progress.set_description(f'{progress_prefix} Epoch {current_epoch}/{self.args.epochs}')

            if scheduled:
                scheduler.step()
            else:
                curr_lr = get_lr(
                    self.lr, self.args.lr_update, epoch, num_epochs,
                    gamma=self.args.gamma, step=self.args.lr_step,
                )
                for param_group in optim.param_groups:
                    param_group['lr'] = curr_lr

            with tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc=f'{progress_prefix} Train', dynamic_ncols=True, leave=False,
            ) as train_progress:
                for i_batch, sample_batched in train_progress:
                    X = (sample_batched['image'] if isinstance(sample_batched, dict) else sample_batched[0]).cuda()
                    y = (sample_batched['label'] if isinstance(sample_batched, dict) else sample_batched[1]).float().cuda()

                    if self._is_diffusion_model:
                        loss, _ = net.train_step({'image': X, 'label': y})
                        loss_terms = {'diffusion': loss.detach()}
                    else:
                        optim.zero_grad()
                        output, aux_out = net(X)
                        loss_main, loss_terms = self.loss_func(output, y, return_details=True)
                        loss_aux, _ = self.loss_func(aux_out, y, return_details=True)
                        loss = loss_main + 0.3 * loss_aux
                        with amp.scale_loss(loss, optimizer) as scaled:
                            scaled.backward()
                        optim.step()

                    train_loss_epoch.append(loss.item())
                    writer.add_scalar('train/loss_total', float(loss.item()), global_step)
                    for key, value in loss_terms.items():
                        writer.add_scalar(f'train/{key}', float(value.detach().item()), global_step)
                    global_step += 1
                    train_progress.set_postfix(
                        loss=f'{float(loss.item()):.4f}',
                        lr=f'{float(optim.param_groups[0]["lr"]):.2e}',
                    )
                    if max_train_batches is not None and (i_batch + 1) >= max_train_batches:
                        break

            mean_train_loss = float(np.mean(train_loss_epoch)) if train_loss_epoch else float('nan')
            should_save_batch_triplet = (current_epoch % save_per_epochs == 0)
            epoch_metrics = {**_nan_epoch, 'train_loss': mean_train_loss, 'val_loss_terms': {}}

            stop_training = False
            if val_loader is not None:
                epoch_metrics, stop_training = run_validation_epoch(
                    solver=self, net=net, model=model, optimizer=optimizer,
                    val_loader=val_loader, epoch=epoch, current_epoch=current_epoch,
                    mean_train_loss=mean_train_loss, save_batch_triplet=should_save_batch_triplet,
                    monitor_metric=monitor_metric, early_stopper=early_stopper,
                )

            if (not save_best_only) and (current_epoch % save_per_epochs == 0):
                torch.save(
                    model.state_dict(),
                    os.path.join(self.args.save, 'models', f'{current_epoch}_model.pth'),
                )
                save_resume_checkpoint(
                    self.args.save, model.state_dict(), optimizer.state_dict(),
                    current_epoch, global_step,
                )

            log_epoch_scalars(
                writer=writer, current_epoch=current_epoch, mean_train_loss=mean_train_loss,
                epoch_metrics=epoch_metrics, early_stopper=early_stopper, val_loader=val_loader,
            )

            epoch_postfix = build_epoch_postfix(
                mean_train_loss=mean_train_loss, val_loader=val_loader,
                epoch_metrics=epoch_metrics, early_stopper=early_stopper, dist_edge_stats=None,
            )
            epoch_progress.set_postfix(epoch_postfix)
            elapsed_hms = format_elapsed_hms(time.perf_counter() - train_start_time)
            write_epoch_result_row(self.args.save, current_epoch, epoch_metrics, elapsed_hms)
            if stop_training:
                break

        epoch_progress.close()
        if completed_epoch == 0:
            raise ValueError('Training loop completed zero epochs; please set --epochs >= 1.')

        final_eval_split = 'test' if test_loader is not None else 'val'
        final_eval_loader = test_loader if test_loader is not None else val_loader
        if final_eval_loader is None:
            raise ValueError('Final evaluation requires test_loader or val_loader.')

        net, final_checkpoint_name = load_best_checkpoint(
            self.args.save, net, val_loader, final_eval_split
        )

        total_elapsed_hms = format_elapsed_hms(time.perf_counter() - train_start_time)
        final_eval_metrics = self.test_epoch(
            net, final_eval_loader, completed_epoch - 1,
            train_loss=epoch_metrics['train_loss'], save_batch_triplet=True,
            split_name=final_eval_split,
        )
        write_eval_summary(
            save_dir=self.args.save, split_name='final', metrics=final_eval_metrics,
            checkpoint_name=final_checkpoint_name, evaluated_split=final_eval_split,
            eval_epoch=completed_epoch, elapsed_hms=total_elapsed_hms,
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
                enumerate(loader), total=len(loader),
                desc=f'{progress_prefix} {split_name.capitalize()} Eval',
                dynamic_ncols=True, leave=False,
            ) as eval_progress:
                for j_batch, test_data in eval_progress:
                    X_test, y_test_raw, sample_names, _ = unpack_test_batch(test_data, j_batch)
                    X_test = X_test.cuda()
                    y_test_raw = y_test_raw.float().cuda()

                    _is_diff = hasattr(model, 'sample')
                    if _is_diff:
                        pred_prob = model.sample(X_test)
                        val_loss = torch.tensor(float('nan'))
                        val_main_dict = {}
                        pred = (pred_prob > 0.5).float()
                    else:
                        output, aux_out = model(X_test)
                        val_loss_main, val_main_dict = self.loss_func(output, y_test_raw, return_details=True)
                        val_loss_aux, _ = self.loss_func(aux_out, y_test_raw, return_details=True)
                        val_loss = val_loss_main + 0.3 * val_loss_aux
                        pred = (torch.sigmoid(output) > 0.5).float()

                    gt_mask = (y_test_raw > 0.5).float()
                    if gt_mask.dim() == 3:
                        gt_mask = gt_mask.unsqueeze(1)

                    if not val_loss.isnan():
                        val_loss_ls.append(val_loss.item())
                    if j_batch == 0:
                        val_term_sums = {k: 0.0 for k in val_main_dict}
                    for k, v in val_main_dict.items():
                        val_term_sums[k] = val_term_sums.get(k, 0.0) + float(v.detach().item())
                    val_term_counts += 1

                    batch = int(X_test.shape[0])
                    dice, Jac = compute_per_class_dice(pred, gt_mask)

                    if self.args.dataset in ('isic', 'cremi'):
                        sample_precision, sample_accuracy = compute_binary_precision_accuracy(pred, gt_mask)
                    else:
                        sample_precision = torch.full((batch,), float('nan'), device=pred.device)
                        sample_accuracy = torch.full((batch,), float('nan'), device=pred.device)
                    precision_ls += sample_precision.detach().cpu().tolist()
                    accuracy_ls += sample_accuracy.detach().cpu().tolist()

                    sample_cldc, sample_betti_error_0, sample_betti_error_1, sample_hd95 = [], [], [], []
                    for b_idx in range(batch):
                        pred_np = pred[b_idx, 0].detach().cpu().numpy()
                        target_np = gt_mask[b_idx, 0].detach().cpu().numpy()
                        cldc = float(clDice(pred_np, target_np))
                        sample_cldc.append(cldc)
                        cldc_ls.append(cldc)
                        betti0_err, betti1_err = getBettiErrors(pred[b_idx, 0], gt_mask[b_idx, 0])
                        sample_betti_error_0.append(float(np.mean(betti0_err)) if betti0_err else float('nan'))
                        sample_betti_error_1.append(float(np.mean(betti1_err)) if betti1_err else float('nan'))
                        betti_error_0_ls.append(sample_betti_error_0[-1])
                        betti_error_1_ls.append(sample_betti_error_1[-1])
                        hd = float(hd95(pred_np, target_np))
                        sample_hd95.append(hd)
                        hd95_ls.append(hd)

                    sample_dice = dice[:, 0]
                    sample_jac = Jac[:, 0]
                    dice_ls += sample_dice.tolist()
                    jac_ls += sample_jac.tolist()

                    if save_batch_triplet:
                        save_checkpoint_batch_triplet(self.args.save, X_test, pred, gt_mask, epoch + 1, j_batch)

                    for b_idx in range(batch):
                        sample_metric_rows.append(build_sample_metric_row(
                            epoch=epoch + 1, batch_idx=j_batch, sample_in_batch=b_idx,
                            sample_name=sample_names[b_idx], val_loss=val_loss.item(),
                            dice=sample_dice[b_idx].item(), jac=sample_jac[b_idx].item(),
                            cldice=sample_cldc[b_idx], precision=sample_precision[b_idx].item(),
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
