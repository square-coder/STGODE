# STGODE
This is an implementation of [Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting](https://arxiv.org/abs/2106.12931) 

## Run
```
python run_stode.py
```

## Requirements
* python 3.7
* torch 1.7.0+cu101
* torchdiffeq 0.2.2
* fastdtw 0.3.4

## Dataset
The datasets used in our paper are collected by the Caltrans Performance Measurement System(PeMS). Please refer to [STSGCN (AAAI2020)](https://github.com/Davidham3/STSGCN) for the download url.

## Reference
Please cite our paper if you use the model in your own work:
```
@inproceedings{fang2021spatial,
  title={Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting},
  author={Fang, Zheng and Long, Qingqing and Song, Guojie and Xie, Kunqing},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={364--373},
  year={2021}
}
```



