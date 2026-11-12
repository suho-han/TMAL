from tqdm.auto import tqdm

from src.utils.monitoring import is_nan_metric
from src.utils.results import save_best_checkpoint


def run_validation_epoch(
    solver,
    net,
    model,
    optimizer,
    val_loader,
    epoch,
    current_epoch,
    mean_train_loss,
    save_batch_triplet,
    monitor_metric,
    early_stopper,
):
    # tqdm.write("RUN VALIDATION ON validation split.")

    epoch_metrics = solver.test_epoch(
        net,
        val_loader,
        epoch,
        train_loss=mean_train_loss,
        save_batch_triplet=save_batch_triplet,
        split_name="val",
    )
    prev_best_dice = float(early_stopper.best_val_dice)
    monitor_value = (
        epoch_metrics["dice"]
        if monitor_metric == "val_dice"
        else epoch_metrics["val_loss"]
    )
    stop_state = early_stopper.step(
        monitor_value=monitor_value,
        val_dice=epoch_metrics["dice"],
        val_loss=epoch_metrics["val_loss"],
        epoch=current_epoch,
    )
    monitor_label = "validation Dice" if monitor_metric == "val_dice" else "validation loss"

    if stop_state["improved"]:
        save_best_checkpoint(
            save_dir=solver.args.save,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            current_epoch=current_epoch,
            stop_state=stop_state,
            monitor_metric=monitor_metric,
            epoch_metrics=epoch_metrics,
        )
        if stop_state["used_tie_break"]:
            tqdm.write(
                "Validation Dice tie detected; lower validation loss selected best checkpoint "
                f'({stop_state["best_val_loss"]:.6f}).'
            )
        else:
            if is_nan_metric(prev_best_dice):
                tqdm.write(
                    "Best model updated based on validation Dice "
                    f'({stop_state["best_val_dice"]:.6f}).'
                )
            else:
                tqdm.write(
                    "Validation Dice improved from "
                    f'{prev_best_dice:.6f} to {stop_state["best_val_dice"]:.6f}. '
                    "Saving best checkpoint."
                )

    stop_training = False
    if stop_state["waiting_for_boundary"]:
        tqdm.write(
            f"Patience reached for {monitor_label}; waiting for stop boundary at every "
            f"{early_stopper.stop_interval} epochs. "
            f"Current epoch: {current_epoch}."
        )
    if stop_state["should_stop"]:
        tqdm.write(
            f"Early stopping triggered. No improvement in {monitor_label} for "
            f"{early_stopper.patience} epochs."
        )
        stop_training = True
    return epoch_metrics, stop_training
