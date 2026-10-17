"""Training entry point for ablation/comparison models (no connectivity supervision)."""
import os

import autorootcwd  # noqa: F401

import torch

from data import build_dataloaders, build_datasets
from engine.train import (
    get_experiment_output_name,
    parse_args,
    save_runtime_config,
    seed_everything,
    seed_worker,
)
from model.registry import build_model as _build_model
from solver_ablation import AblationSolver
from utils import create_exp_directory


def main(args):
    torch.cuda.set_device(args.device)
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
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
        model = model.cuda()
        print("Loaded pretrained model from %s" % args.pretrained)

    solver = AblationSolver(args)
    solver.train(
        model, train_loader, val_loader,
        num_epochs=args.epochs, label_mode=args.label_mode,
        test_loader=test_loader,
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
