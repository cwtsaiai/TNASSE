# A Training-free Neural Architecture Search Based on Search Economics (TNASSE) [PDF](https://www.doi.org/10.1109/TEVC.2023.3264533)

## Overview
Coming soon...

## Installation

Clone this repo:
```
git clone https://github.com/cwtsaiai/TNASSE.git
```

We provide a conda environment setup file. Create a conda environment `tnasse` by running:
```
conda env create -f env.yml
```
Activate the environment: 
```
conda activate tnasse
```

Benchmark datasets used in this study:
1. NAS-Bench-101
Download [nasbench_only108.tfrecord](https://github.com/google-research/nasbench) file and place it in searchspace folder.

2. NAS-Bench-201
Download [NAS-Bench-201-v1_1-096897.pth](https://github.com/D-X-Y/NAS-Bench-201) file and place it in searchspace folder.

3. NATS-Bench-SSS
Download [NATS-sss-v1_0-50262-simple.tar](https://github.com/D-X-Y/NATS-Bench) file and place it in searchspace folder.

## Usage

Here we provide a script to reproduce the results
```
./search.sh
```
## Citation
 [Meng-Ting Wu](),  [Hung-I Lin](), [Chun-Wei Tsai](https://sites.google.com/site/cwtsai0807/chun-wei-tsai),
 ["A Training-Free Neural Architecture Search Algorithm Based on Search Economics"](https://www.doi.org/10.1109/TEVC.2023.3264533), <i>IEEE Transactions on Evolutionary Computation (TEVC)</i>, 2023, In Press.

```
@ARTICLE{Wu-2023,
  author={Wu, Meng-Ting and Lin, Hung-I and Tsai, Chun-Wei},
  journal={IEEE Transactions on Evolutionary Computation},
  title={A Training-Free Neural Architecture Search Algorithm Based on Search Economics},
  doi={10.1109/TEVC.2023.3264533},
  year={2023, In Press}
}
```
