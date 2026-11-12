import torch


def normalize_sample_names(raw_name, batch_size, batch_idx):
    if raw_name is None:
        return [f"batch_{batch_idx:04d}_idx_{i:02d}" for i in range(batch_size)]
    if isinstance(raw_name, str):
        return [raw_name] if batch_size == 1 else [f"{raw_name}_{i}" for i in range(batch_size)]
    if torch.is_tensor(raw_name):
        names = [str(v) for v in raw_name.detach().cpu().view(-1).tolist()]
    elif isinstance(raw_name, (list, tuple)):
        names = [str(v) for v in raw_name]
    else:
        return [str(raw_name)] if batch_size == 1 else [f"{str(raw_name)}_{i}" for i in range(batch_size)]
    if len(names) < batch_size:
        names += [f"batch_{batch_idx:04d}_idx_{i:02d}" for i in range(len(names), batch_size)]
    return names[:batch_size]


def unpack_test_batch(test_data, batch_idx):
    """Unpack a DataLoader batch to (X, y, sample_names, binary_gt)."""
    binary_gt = None
    if isinstance(test_data, dict):
        X = test_data['image']
        y = test_data['label']
        binary_gt = test_data.get('binary_gt', None)
        name_src = test_data.get('name', None)
    elif isinstance(test_data, (list, tuple)):
        if len(test_data) < 2:
            raise ValueError('test_data must contain at least image and label tensors')
        if len(test_data) >= 3 and not torch.is_tensor(test_data[0]) and torch.is_tensor(test_data[1]):
            X, y, name_src = test_data[1], test_data[2], test_data[0]
        else:
            X, y = test_data[0], test_data[1]
            name_src = test_data[2] if len(test_data) >= 3 else None
    else:
        raise TypeError('test_data must be a dict or tuple/list from DataLoader')
    if not torch.is_tensor(X) or not torch.is_tensor(y):
        raise TypeError('image and label in test_data must be tensors')
    if binary_gt is not None:
        binary_gt = binary_gt.cuda()
    batch_size = int(X.shape[0])
    sample_names = normalize_sample_names(name_src, batch_size, batch_idx)
    return X, y, sample_names, binary_gt
