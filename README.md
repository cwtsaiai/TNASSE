# A Training-Free Neural Architecture Search Algorithm Based on Search Economics (TNASSE) [PDF](https://www.doi.org/10.1109/TEVC.2023.3264533)

This is the companion code for the IEEE TEVC paper: [A Training-Free Neural Architecture Search Algorithm Based on Search Economics](https://www.doi.org/10.1109/TEVC.2023.3264533)

If you have any questions regarding the paper or encounter any issues while attempting to reproduce the results, please do not hesitate to contact us via email or by opening an issue. We will be happy to assist you.

## Overview

This repository contains an implementation of an efficient Neural Architecture Search (NAS) algorithm that is based on an improved version of the Search Economics (SE) metaheuristic algorithm and a novel training-free score function.

The proposed NAS algorithm uses the expected value of each region in the search space to guide the search, enabling it to focus on high-potential regions and significantly reduce computation time. We aim to overcome the limitations of existing training-free methods that use simple metaheuristic algorithms and score functions that may misjudge the quality of a neural architecture.
Experimental results show that the proposed NAS algorithm can find a similar or better result than most non-training-free NAS algorithms, but with a much shorter computation time. The article's contributions can be summarized as a more efficient and accurate method for NAS.

For more details, please refer to our [paper](https://www.doi.org/10.1109/TEVC.2023.3264533).

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
1. NAS-Bench-101:
Download [nasbench_only108.tfrecord](https://github.com/google-research/nasbench) file and place it in searchspace folder.

2. NAS-Bench-201:
Download [NAS-Bench-201-v1_1-096897.pth](https://github.com/D-X-Y/NAS-Bench-201) file and place it in searchspace folder.

3. NATS-Bench-SSS:
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
