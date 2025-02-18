# EarthNetV1

Code for the paper: ["Global atmospheric data assimilation with multi-modal masked autoencoders"](https://arxiv.org/abs/2407.11696), developed by Zeus AI Inc. 

## Setup environment with miniconda

```
conda env create -f environment.yml
```


## Download data and model weights

Test data is included from the subset of available MIRS data.  A larger dataset was used to pre-train with GEO, ATMS, and VIIRS alone.

```
mkdir data checkpoints
aws s3 cp --endpoint https://fly.storage.tigris.dev --no-sign-request s3://zeus-public/earthnet/v1/GEO-ATMS-VIIRS-MIRS.test.tar data/
tar -xvf GEO-ATMS-VIIRS-MIRS.tar
aws s3 cp --endpoint https://fly.storage.tigris.dev --no-sign-request s3://zeus-public/earthnet/v1/earthnet.v1.ckpt checkpoints/
```

`demo.ipynb` and `inference_earthnet.py` show example of how to make predictions with EarthNet. 

## Cite us

```
@article{vandal2024global,
  title={Global atmospheric data assimilation with multi-modal masked autoencoders},
  author={Vandal, Thomas J and Duffy, Kate and McDuff, Daniel and Nachmany, Yoni and Hartshorn, Chris},
  journal={arXiv preprint arXiv:2407.11696},
  year={2024}
}
```
