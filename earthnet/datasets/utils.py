import numpy as np
import dask as da
import torch
from torch.utils import data
import os
import pickle


def interp_dim(x, scale):
    x0, xlast = x[0], x[-1]
    newlength = int(len(x) * scale)
    y = np.linspace(x0, xlast, num=newlength, endpoint=False)
    return y


def blocks(data, width=352):
    # n = data.t.shape[0]
    w = data.x.shape[0]
    h = data.y.shape[0]

    hs = np.arange(0, h, width)
    ws = np.arange(0, w, width)
    blocks = []
    for hindex in hs:
        if hindex + width > h:
            hindex = h - width

        for windex in ws:
            if windex + width > w:
                windex = w - width
            blocks.append(
                data.sel(
                    y=data.y.values[hindex : hindex + width],
                    x=data.x.values[windex : windex + width],
                )
            )
    return blocks


def block_dask_array(arr, axis, size=128, stride=128):
    arr = da.array.swapaxes(arr, axis, 0)
    n = arr.shape[0]
    stack = []
    for j in range(0, n, stride):
        j = min(j, n - size)
        stack.append(arr[j : j + size])
    stack = da.array.stack(stack)
    stack = da.array.swapaxes(stack, axis + 1, 1)
    return stack


def block_array(arr, axis, size=128, stride=128):
    arr = np.swapaxes(arr, axis, 0)
    n = arr.shape[0]
    stack = []
    for j in range(0, n, stride):
        j = min(j, n - size)
        stack.append(arr[np.newaxis, j : j + size])
    stack = np.concatenate(stack, 0)
    stack = np.swapaxes(stack, axis + 1, 1)
    return stack


def xarray_to_block_list(arr, dim, size=128, stride=128):
    n = arr[dim].shape[0]
    stack = []
    for j in range(0, n, stride):
        j = min(j, n - size)
        stack.append(arr.isel({dim: np.arange(j, j + size)}))
    return stack


def interp(da, scale, fillna=False):
    xnew = interp_dim(da["x"].values, scale)
    ynew = interp_dim(da["y"].values, scale)
    newcoords = dict(x=xnew, y=ynew)
    return da.interp(newcoords)


def regrid_2km(da, band):
    if band == 2:
        return interp(da, 1.0 / 4, fillna=False)
    elif band in [1, 3, 5]:
        return interp(da, 1.0 / 2, fillna=False)
    return da


def regrid_1km(da, band):
    if band == 2:  # (0.5 km)
        return interp(da, 1.0 / 2, fillna=False)
    elif band not in [1, 3, 5]:  # 2km
        return interp(da, 2.0, fillna=False)
    return da


def regrid_500m(da, band):
    if band == 2:  # 500m
        return da
    elif band in [1, 3, 5]:  # 1km
        return interp(da, 2.0, fillna=False)
    return interp(da, 4.0, fillna=False)  # 2km


def cartesian_to_speed(da):
    lat_rad = np.radians(da.lat.values)
    lon_rad = np.radians(da.lon.values)
    a = np.cos(lat_rad) ** 2 * np.sin((lon_rad[1] - lon_rad[0]) / 2) ** 2
    d = 2 * 6378.137 * np.arcsin(a**0.5)
    size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1)  # km
    da["U"] = da["U"] * size_per_pixel * 1000 / 1800
    da["V"] = da["V"] * size_per_pixel * 1000 / 1800
    return da


def speed_to_cartesian(da):
    lat_rad = np.radians(da.lat.values)
    lon_rad = np.radians(da.lon.values)
    a = np.cos(lat_rad) ** 2 * np.sin((lon_rad[1] - lon_rad[0]) / 2) ** 2
    d = 2 * 6378.137 * np.arcsin(a**0.5)
    size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1)  # km
    da["U"] = da["U"] / size_per_pixel / 1000 * 1800 / 0.9
    da["V"] = da["U"] / size_per_pixel / 1000 * 1800 / 0.9
    return da


class SparseDataLoader(data.DataLoader):
    """
    A PyTorch Sampler for sparse datasets that returns only valid samples, according to a user-specified criterion. The loader
    caches the indices of valid and invalid samples to speed up data loading by avoiding re-checking invalid samples.

    Parameters:
    ----------
        dataset: a PyTorch Dataset object
        criterion (function): A function to evaluate samples for validity (default: lambda x: np.sum(~np.isnan(x)) / np.prod(x.shape) > 0.01).
        **kwargs: Additional keyword arguments for DataLoader.

    """


class SparseSampler(data.Sampler):
    """
    A PyTorch Sampler for sparse datasets that returns only valid samples, according to a user-specified criterion. The sampler
    caches the indices of valid and invalid samples to speed up data loading by avoiding re-checking invalid samples.

    Parameters:
    ----------
        dataset: a PyTorch Dataset object
        shuffle (bool): Whether to shuffle the indices (default: True).
        cache_dir (str): The directory to save the indices cache (default: 'dataloader_cache').
        criterion (function): A function to evaluate samples for validity (default: lambda x: np.sum(~np.isnan(x)) / np.prod(x.shape) > 0.01).
        **kwargs: Additional keyword arguments for DataLoader.

    """

    def __init__(
        self,
        dataset,
        shuffle=True,
        cache_dir="dataloader_cache",
        criterion=lambda x: np.sum(~np.isnan(x)) / np.prod(x.shape) > 0.1,
    ):
        self.dataset = dataset
        self.criterion = criterion
        self.cache_dir = cache_dir
        self.indices = self._initialize_indices()
        self.shuffle = shuffle

    def _initialize_indices(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "indices.pkl")
        indices = (
            pickle.load(open(self.cache_file, "rb"))
            if os.path.exists(self.cache_file)
            else np.zeros(len(self.dataset))
        )
        return indices

    def _cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        pickle.dump(self.indices, open(self.cache_file, "wb"))

    def _update_list(self, idx, flag):
        self.indices[idx] = flag
        self._cache()

    def _find_valid_sample(self):
        while len(np.where(self.indices == 0)[0]) > 0:
            idx = np.random.choice(np.where(self.indices == 0)[0])
            x = self.dataset[int(idx)]
            if self.criterion(x):
                self._update_list(idx, 1)
                return idx
            else:
                self._update_list(idx, -1)

        valid_examples = np.where(self.indices == 1)[0]
        idx = np.random.choice(valid_examples)
        return int(idx)

    def _check_sample(self, idx):
        flag = self.indices[idx]
        if flag == 1:
            idx = idx
        elif flag == -1:
            idx = self._find_valid_sample()
        elif flag == 0:
            idx = self._find_valid_sample()
        return idx

    def __iter__(self):

        if len(np.where(self.indices == 0)[0]) == 0:
            valid_indices = np.where(self.indices == 1)[0]
            if self.shuffle:
                valid_indices = np.random.permutation(valid_indices)
            for i in valid_indices:
                yield int(i)

        else:
            for i in torch.randperm(len(self.indices)):
                i = self._check_sample(i)
                yield int(i)

    def __len__(self) -> int:
        return len(self.indices)
