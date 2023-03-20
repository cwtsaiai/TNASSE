# A Training-Free Neural Architecture Search Algorithm based on Search Economics
### [Paper]()

 [A Training-Free Neural Architecture Search Algorithm based on Search Economics]() <br>
 [Meng-Ting Wu](),
 [Hun-I Lin](),
 [Chun-Wei Tsai](https://sites.google.com/site/cwtsai0807/chun-wei-tsai) <br>
  National Sun Yat-sen University  
in IEEE Transactions on Evolutionary Computation (TEVC)

## Summary


## Download

### NAS-Bench-101
download [nasbench_only108.tfrecord](https://github.com/google-research/nasbench) file and place it in searchspace folder.

### NAS-Bench-201
Download [NAS-Bench-201-v1_1-096897.pth](https://github.com/D-X-Y/NAS-Bench-201) file and place it in searchspace folder.

### NATS-Bench-SSS
Download [NATS-sss-v1_0-50262-simple.tar](https://github.com/D-X-Y/NATS-Bench) file and place it in searchspace folder.
## Setup

We provide a conda environment setup file. Create a conda environment `tnasse` by running:
```
conda env create -f env.yml
```
Activate the environment 
```
conda activate tnasse
```

## Usage

Here we provide a script to reproduce the results
```
./search.sh
```

## Citation

```
@article{wu2023tnasse,
  title={A Training-Free Neural Architecture Search Algorithm based on Search Economics},
  author={Meng-Ting Wu and Hung-I Lin and Chun-Wei Tsai},
  year={2023},
  booktitle={IEEE Transactions on Evolutionary Computation},
}
```