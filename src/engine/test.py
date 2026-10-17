import os

from src.utils import write_epoch_result_row, write_eval_summary, write_result_txt


def run_test_only_eval(solver, net, val_loader, test_loader, writer):
    eval_split_name = "test" if test_loader is not None else "val"
    print(f"START {eval_split_name.upper()}-ONLY EVAL.")
    test_metrics = solver.test_epoch(
        net,
        val_loader if test_loader is None else test_loader,
        0,
        split_name=eval_split_name,
        save_batch_triplet=True,
    )
    write_epoch_result_row(solver.args.save, 1, test_metrics, elapsed_hms="")
    checkpoint_name = os.path.basename(solver.args.pretrained) if solver.args.pretrained else ""
    write_eval_summary(
        save_dir=solver.args.save,
        split_name="final",
        metrics=test_metrics,
        checkpoint_name=checkpoint_name,
        evaluated_split=eval_split_name,
        eval_epoch=1,
        elapsed_hms="",
    )
    write_result_txt(
        save_dir=solver.args.save,
        dataset=solver.args.dataset,
        split_name="final",
        metrics=test_metrics,
        checkpoint_name=checkpoint_name,
        evaluated_split=eval_split_name,
        eval_epoch=1,
        elapsed_hms="",
    )
    writer.close()
