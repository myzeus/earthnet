import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseTransform(pl.LightningModule):
    def __init__(self, keys=None, ckpt_path=None, **kwargs):
        super().__init__()
        self.checkpoint_path = ckpt_path
        self.keys = keys

    def forward(self, sample):
        """
        Sample is a dict with any number of keys.
        """
        return sample

    def backward(self, y):
        return y

    def training_step(self, batch, batch_idx):
        return 0

    def validation_step(self, batch, batch_idx):
        return 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


class StandardScalar(BaseTransform):
    def __init__(self, keys, dim=1, keys_dim=None, **kwargs):
        """
        Normalize features into uniform distribution with step size `step`.
        Compute mapping from a large training batch with mappings per feature.
        """
        super().__init__(keys=keys, **kwargs)
        self.dim = dim
        self.mu = nn.ParameterDict(
            {
                k: nn.parameter.Parameter(torch.zeros(d), requires_grad=False)
                for k, d in self.keys.items()
            }
        )
        self.sd = nn.ParameterDict(
            {
                k: nn.parameter.Parameter(torch.ones(d), requires_grad=False)
                for k, d in self.keys.items()
            }
        )

    def forward(self, sample):
        for k in sample.keys():  # self.keys:
            if k not in self.keys:
                print("Transform Key {} not in keys".format(k))
                continue
            sample[k] = sample[k].swapaxes(-1, self.dim)
            sample[k] = (sample[k] - self.mu[k]) / self.sd[k]
            sample[k] = sample[k].swapaxes(-1, self.dim)

        return sample

    def backward(self, batch):
        y = dict()
        for k in batch.keys():  # self.keys:
            x = batch[k]
            x = x.swapaxes(-1, self.dim)
            x = (x * self.sd[k]) + self.mu[k]
            x = x.swapaxes(-1, self.dim)
            y[k] = x
        return y

    def training_step(self, batch, batch_idx):

        for k in self.keys:
            x = batch[k]
            x = x.swapaxes(0, self.dim)
            x = x.reshape(x.shape[0], -1)
            self.mu[k] = nn.parameter.Parameter(x.nanmean(dim=1), requires_grad=False)
            self.sd[k] = nn.parameter.Parameter(nanstd(x, dim=1), requires_grad=False)

        # self.mu = nn.ParameterDict(self.mu)

    # self.sd = nn.ParameterDict(self.sd)


class QuantileTransform(BaseTransform):
    def __init__(self, keys, n_quantiles=1000, dim=1, dist_name="uniform", **kwargs):
        """
        Normalize features into uniform distribution with step size `step`.
        Compute mapping from a large training batch with mappings per feature.
        """
        super().__init__(keys, **kwargs)
        self.dim = dim
        self.step = 1 / n_quantiles
        self.dist_name = dist_name
        if dist_name == "uniform":
            self.dist = torch.distributions.uniform.Uniform(0, 1)
        elif dist_name == "normal":
            self.dist = torch.distributions.normal.Normal(0, 1)

        # self.x_bins = self.dist.icdf(cdf).contiguous()
        self.x_bins = torch.arange(self.step / 2, 1, self.step)
        self.quantiles = nn.ParameterDict(
            {
                k: nn.parameter.Parameter(torch.zeros([n_quantiles, d]))
                for k, d in self.keys.items()
            }
        )

    def forward(self, sample, max_search=1000):
        y = dict()
        self.x_bins = self.x_bins.to(sample[list(sample.keys())[0]].device)
        for k in sample:
            if k not in self.keys:
                continue
            x = sample[k].contiguous()
            self.quantiles[k] = self.quantiles[k].contiguous()
            x = x.swapaxes(0, self.dim)
            x_shape = x.shape

            x = x.reshape(x.shape[0], -1)
            y_k = torch.zeros_like(x)

            for i in range(x.shape[0]):
                for j in range(0, x.shape[1], max_search):
                    y_k_i_idxs = torch.searchsorted(
                        self.quantiles[k][:, i], x[i, j : j + max_search]
                    )
                    y_k_i_idxs[y_k_i_idxs == self.quantiles[k].shape[0]] -= 1
                    y_k[i, j : j + max_search] = self.x_bins[y_k_i_idxs]

            y[k] = y_k.reshape(x_shape)
            y[k] = y[k].swapaxes(0, self.dim)
            # y[k][y[k] == 0] = self.step / 2
            y[k] = self.dist.icdf(y[k])

        return y

    def backward(self, batch):
        y = dict()
        for k in batch:
            x = batch[k]
            if self.dist_name == "uniform":
                x[x < 0] = 0
                x[x > 1] = 1
            x = self.dist.cdf(x)
            x = x.swapaxes(0, self.dim)
            x_shape = x.shape

            x = x.reshape(x.shape[0], -1)

            y[k] = torch.zeros_like(x)
            for i in range(x.shape[0]):
                x_i_idxs = torch.searchsorted(self.x_bins, x[i])
                x_i_idxs[x_i_idxs == self.x_bins.shape[0]] -= 1
                y[k][i] = self.quantiles[k][x_i_idxs, i]

            y[k] = y[k].reshape(x_shape)
            y[k] = y[k].swapaxes(0, self.dim)

        return y

    def training_step(self, batch, batch_idx):
        for k in batch:
            x = batch[k]
            x = x.swapaxes(0, self.dim)
            x = x.reshape(x.shape[0], -1)
            self.quantiles[k] = torch.nanquantile(x, self.x_bins, dim=1).contiguous()


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output
