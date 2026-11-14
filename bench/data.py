"""Data loaded straight onto the GPU, with augmentation done on the GPU.

No DataLoader/workers: faster for small images, avoids worker issues on
Windows, and every random operation (shuffle, crop, flip) draws from one
CPU torch.Generator re-seeded from (seed, epoch). An interrupted and resumed
run therefore gives exactly the same result as an uninterrupted one.
"""
import torch

MEAN_STD = {
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
IN_CHANNELS = {"fmnist": 1, "cifar10": 3, "cifar100": 3}
NUM_CLASSES = {"fmnist": 10, "cifar10": 10, "cifar100": 100}
N_TRAIN_FULL = {"fmnist": 60000, "cifar10": 50000, "cifar100": 50000}


def _load_raw(dataset, root):
    import torchvision
    if dataset == "fmnist":
        tr = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        te = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        xtr, ytr = tr.data.unsqueeze(1), torch.as_tensor(tr.targets)
        xte, yte = te.data.unsqueeze(1), torch.as_tensor(te.targets)
    elif dataset in ("cifar10", "cifar100"):
        cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
        tr = cls(root, train=True, download=True)
        te = cls(root, train=False, download=True)
        xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
        xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
        ytr, yte = torch.as_tensor(tr.targets), torch.as_tensor(te.targets)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return xtr, ytr.long(), xte, yte.long()


def _fake_raw(dataset, n_train=1200, n_test=300):
    """Fake data (for --selftest without network access). Structured so that the model can learn."""
    c = IN_CHANNELS[dataset]
    s = 28 if dataset == "fmnist" else 32
    k = NUM_CLASSES[dataset]
    g = torch.Generator().manual_seed(0)
    base = torch.randint(0, 256, (k, c, s, s), generator=g).float()

    def make(n):
        y = torch.randint(0, k, (n,), generator=g)
        noise = torch.randn(n, c, s, s, generator=g) * 60
        x = (base[y] + noise).clamp(0, 255).to(torch.uint8)
        return x, y
    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte


class GPUData:
    """Keeps train/val/test as uint8 on the device; normalizes per batch."""

    def __init__(self, dataset, root, device, val_size, split_seed,
                 fake=False, train_limit=None):
        self.dataset = dataset
        self.device = device
        self.augment = dataset in ("cifar10", "cifar100")
        xtr, ytr, xte, yte = _fake_raw(dataset) if fake else _load_raw(dataset, root)

        if fake:
            val_size = min(val_size, len(xtr) // 6)
        perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(split_seed))
        vidx, tidx = perm[:val_size], perm[val_size:]
        if train_limit is not None:
            tidx = tidx[:train_limit]

        self.x_train, self.y_train = xtr[tidx].to(device), ytr[tidx].to(device)
        self.x_val, self.y_val = xtr[vidx].to(device), ytr[vidx].to(device)
        self.x_test, self.y_test = xte.to(device), yte.to(device)

        mean, std = MEAN_STD[dataset]
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)

    @property
    def n_train(self):
        return len(self.x_train)

    def _norm(self, x_uint8):
        return (x_uint8.float().div_(255.0) - self.mean) / self.std

    def _crop_flip(self, x, gen):
        """RandomCrop(32, padding=4) + RandomHorizontalFlip, vectorized.
        Zero padding on [0,1] images before normalization, as in torchvision."""
        n, c, h, w = x.shape
        pad = 4
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        oy = torch.randint(0, 2 * pad + 1, (n,), generator=gen).to(x.device)
        ox = torch.randint(0, 2 * pad + 1, (n,), generator=gen).to(x.device)
        flip = (torch.rand(n, generator=gen) < 0.5).to(x.device)
        ar_h = torch.arange(h, device=x.device)
        ar_w = torch.arange(w, device=x.device)
        rows = (oy[:, None] + ar_h[None, :])[:, None, :, None]
        cols = (ox[:, None] + ar_w[None, :])[:, None, None, :]
        bi = torch.arange(n, device=x.device)[:, None, None, None]
        ci = torch.arange(c, device=x.device)[None, :, None, None]
        out = xp[bi, ci, rows, cols]
        return torch.where(flip[:, None, None, None], out.flip(3), out)

    def train_batches(self, batch_size, seed, epoch):
        gen = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
        perm = torch.randperm(self.n_train, generator=gen).to(self.device)
        for i in range(0, self.n_train, batch_size):
            idx = perm[i:i + batch_size]
            x = self.x_train[idx].float().div_(255.0)
            if self.augment:
                x = self._crop_flip(x, gen)
            x = (x - self.mean) / self.std
            yield x, self.y_train[idx]

    def eval_batches(self, split, batch_size=1000):
        x_all = self.x_val if split == "val" else self.x_test
        y_all = self.y_val if split == "val" else self.y_test
        for i in range(0, len(x_all), batch_size):
            yield self._norm(x_all[i:i + batch_size]), y_all[i:i + batch_size]

    def steps_per_epoch(self, batch_size):
        return (self.n_train + batch_size - 1) // batch_size
