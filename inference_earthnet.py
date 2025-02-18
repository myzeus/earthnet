import glob
import xarray as xr
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from earthnet.util import instantiate_from_config


class EarthNetRunner(object):
    def __init__(self, config_file):
        self.config = OmegaConf.load(config_file)
        self.model = instantiate_from_config(self.config.model)

    def prepare_inputs(self, data):
        assert len(data.time) == 12

        # del data["PIce"]
        for v in data.data_vars:
            data[v] = data[v].astype("float16")
        # data.to_zarr("earthmae_abi-viirs-atms_inputs.zarr")

        # data = data.isel(lat=slice(0,500), lon=slice(0,500))
        inputs = dict()
        for d, params in self.config.data.train.params.domains.items():
            inputs[d] = []
            for v in params["vars"]:
                key = d + "_" + v
                if len(data[key].dims) == 4:
                    c_dim = np.setdiff1d(data[key].dims, ["time", "lat", "lon"])[0]
                    data[key] = data[key].transpose(c_dim, "time", "lat", "lon")
                elif len(data[key].dims) == 3:
                    if "time" in data[key].dims:
                        new_dim = "band_" + v
                        data[key] = data[key].expand_dims(new_dim)
                elif (len(data[key].dims) == 2) and (
                    "time" not in data[key].dims
                ):  # make 2d sample (C, H, W)
                    new_dim = "band_" + v
                    # x[key] = x[key].expand_dims("time_0")
                    data[key] = data[key].expand_dims(new_dim)

                inputs[d].append(torch.Tensor(data[key].values))

            inputs[d] = torch.cat(
                inputs[d]
            )  # torch.cat([torch.Tensor(data[v].values) for v in params["vars"]], 1)

            inputs[d] = inputs[d].unsqueeze(0)
        return inputs

    def forward(self, x: dict) -> dict:
        """
        Args:
            x: dict of modalities. eg. {'goes16': torch.Tensor(), 'viirs': ....}

        Returns:
            Dict of gap-filled modalities
        """
        patch_size = self.config.model.params.img_size
        patch_overlap = patch_size // 4
        trim = 0

        device = self.model.device

        # x = self.model.data_transform(x)

        # perform inference on patches
        counters = {k: np.zeros(v.shape, dtype=np.float32) for k, v in x.items()}
        res_sum = {k: np.zeros(v.shape, dtype=np.float32) for k, v in x.items()}
        if "goes16" in x.keys():
            height, width = x["goes16"].shape[-2:]
        else:
            height, width = x[list(x.keys())[0]].shape[-2:]

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7067425/
        # hann = 1/4 * (1 - cos(2 \pi i / I)) * (1 - cos(2 \pi j / J))
        xr = np.arange(0, patch_size, dtype=np.float32)
        xr_w = 1 - np.cos(2 * np.pi * xr / patch_size)
        pdf = np.outer(xr_w, xr_w) / 4

        border_mask = np.zeros(pdf.shape)
        if trim == 0:
            border_mask = 1.0
        else:
            border_mask[trim:-trim, trim:-trim] = 1
        pdf *= border_mask

        indices = []
        ix_iy = [
            (ix, iy)
            for ix in range(0, height, patch_size - patch_overlap)
            for iy in range(0, width, patch_size - patch_overlap)
        ]
        for ix, iy in tqdm(ix_iy):
            ix = min(ix, height - patch_size)
            iy = min(iy, width - patch_size)
            if (ix, iy) in indices:
                continue
            indices.append((ix, iy))
            patch_inputs = {
                k: v[..., ix : ix + patch_size, iy : iy + patch_size].to(device)
                for k, v in x.items()
            }

            patch_inputs = self.model.data_transform(patch_inputs)

            patch_outputs = self.model(patch_inputs)[0]
            # patch_outputs = self.model(patch_inputs, mask_inputs=True)[0]
            patch_outputs = self.model.backward_transform(patch_outputs)

            for k, v in patch_outputs.items():
                res_sum[k][..., ix : ix + patch_size, iy : iy + patch_size] += (
                    v.cpu().detach().numpy() * pdf
                )
                counters[k][..., ix : ix + patch_size, iy : iy + patch_size] += pdf

            del patch_inputs

        out = {}
        for k, v in res_sum.items():
            out[k] = v / counters[k]  # .compute()
            # print(k, out[k].shape)

        return out

    def outputs_to_dataset(self, ds: xr.Dataset, outputs: dict) -> xr.Dataset:
        """
        Returns a gap-filled dataset like ds
        """
        pred = xr.full_like(ds, np.nan)
        for d, params in self.config.data.train.params.domains.items():
            idx = 0
            for v in params["vars"]:
                k = d + "_" + v

                dim_diff = np.setdiff1d(pred[k].dims, ["time", "lat", "lon"])
                if len(dim_diff) == 1:
                    idx_end = pred[dim_diff[0]].shape[0] + idx
                    pred[k] = pred[k].transpose(
                        dim_diff[0], "time", "lat", "lon", missing_dims="ignore"
                    )
                    pred[k].values = outputs[d][0, idx:idx_end]
                    idx = idx_end
                elif len(dim_diff) == 0:
                    pred[k].values = outputs[d][0, idx]
                    idx += 1
                else:
                    raise ValueError()

                # print(k, idx, idx_end, dim_diff)

                pred[k] = pred[k].astype("float16")
                idx_end = idx

        return pred


if __name__ == "__main__":
    config_file = "multimodal_earth.yaml"

    config = OmegaConf.load(config_file)

    runner = EarthNetRunner(config_file)

    print("Loading data")
    data_files = glob.glob(config.data.test.params.data_dir + "*.zarr")
    data = xr.open_zarr(data_files[0])
    data = data.isel(time=slice(0, 12))

    print("Running model")
    inputs = runner.prepare_inputs(data)
    outputs = runner.forward(inputs)
    pred = runner.outputs_to_dataset(data, outputs)

    print("Prediction\n", pred)

    pred.to_zarr("earthnet_predictions.zarr")
