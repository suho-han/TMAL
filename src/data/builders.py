import torch

from data_loader.GetDataset_DRIVE import MyDataset_DRIVE
from data_loader.GetDataset_DeepCrack import MyDataset_DeepCrack


def build_datasets(args):
    validset = None
    testset = None

    if args.dataset == "drive":
        trainset = MyDataset_DRIVE(
            args,
            train_root=args.data_root,
            mode="train",
            label_mode=args.label_mode,
        )
        testset = MyDataset_DRIVE(
            args,
            train_root=args.data_root,
            mode="test",
            label_mode=args.label_mode,
        )
    elif args.dataset == "deepcrack":
        trainset = MyDataset_DeepCrack(
            args,
            train_root=args.data_root,
            mode="train",
            label_mode=args.label_mode,
        )
        testset = MyDataset_DeepCrack(
            args,
            train_root=args.data_root,
            mode="test",
            label_mode=args.label_mode,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return trainset, validset, testset


def build_dataloaders(args, trainset, validset, testset, seed_worker, loader_generator):
    num_workers = int(getattr(args, 'num_workers', 8))
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_loader = None
    if validset is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset=validset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )

    test_loader = None
    if testset is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
    elif val_loader is not None:
        # Backward-compatible fallback for datasets with a single held-out split.
        test_loader = val_loader

    return train_loader, val_loader, test_loader
