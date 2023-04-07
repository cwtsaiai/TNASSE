# Source Codes of Training-free Neural Architecture Search based on Search Economics (TNASSE)
<b>We are cleaning up the source codes now, and all of them will be uploaded a few days later<\b>

## 1. A Brief Introduction to TNASSE

## 2. Download
### 2.1 Datasets
#### NAS-Bench-101
Download [nasbench_only108.tfrecord](https://github.com/google-research/nasbench) file and place it in searchspace folder.

#### NAS-Bench-201
Download [NAS-Bench-201-v1_1-096897.pth](https://github.com/D-X-Y/NAS-Bench-201) file and place it in searchspace folder.

#### NATS-Bench-SSS
Download [NATS-sss-v1_0-50262-simple.tar](https://github.com/D-X-Y/NATS-Bench) file and place it in searchspace folder.
### Setup

We provide a conda environment setup file. Create a conda environment `tnasse` by running:
```
conda env create -f env.yml
```
Activate the environment 
```
conda activate tnasse
```
### 2.2 Source Coees

### 2.3 Usage

Here we provide a script to reproduce the results
```
./search.sh
```
## 3. Paper Information
 [Meng-Ting Wu](),  [Hung-I Lin](), [Chun-Wei Tsai](https://sites.google.com/site/cwtsai0807/chun-wei-tsai),
 ["A Training-Free Neural Architecture Search Algorithm based on Search Economics"](https://ieeexplore.ieee.org/document/10092788), <i>IEEE Transactions on Evolutionary Computation (TEVC)</i>, 2023, In Press.

Here is a BiBTeX citation as well: <br>
@ARTICLE{Wu-2023, <br>
  author={Wu, Meng-Ting and Lin, Hung-I and Tsai, Chun-Wei}, <br>
  journal={IEEE Transactions on Evolutionary Computation},  <br>
  title={A Training-Free Neural Architecture Search Algorithm based on Search Economics},  <br>
  doi={10.1109/TEVC.2023.3264533}, <br>
  year={2023, In Press} <br>
}
