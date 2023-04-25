import torch
import numpy as np
import random
import pandas as pd
import argparse
import importlib
from statistics import mean
import re
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed',default=None,type=int,help='random seed')
parser.add_argument('--algo',default='se',type=str,help='algorithm')
parser.add_argument('--score_function',default='ni',type=str,help='score function (ni,...)')
parser.add_argument('--sigma',default=1,type=float,help='standard deviation of gaussian noise')
parser.add_argument('--encoding',default='backbone',type=str,help='encoding scheme (hash,backbone)')
parser.add_argument('--runs',default=50,type=int,help='number of runs')
parser.add_argument('--iters',default=1000,type=int,help='number of iterations for each run')
parser.add_argument('--sl',default=21,type=int,help='sequence length')
parser.add_argument('--ptype',default='nasbench201',type=str,help='problem type (nasbench101,nasbench201,etc.)')
parser.add_argument('--atom',default=None,type=int,help='number of choices per node/edge')
parser.add_argument('--max_evaluations',default=1000,type=int,help='number of evaluations for each run')
#----------------------------------------------------SE parameters--------------------------------------------------------#
parser.add_argument('--n',default=4,type=int,help='number of searchers')
parser.add_argument('--h',default=4,type=int,help='number of regions (SE)')
parser.add_argument('--w',default=2,type=int,help='the number of possible goods (the number of samples of each region)  (SE)')
#----------------------------------------------------NAS-BENCH-201 & NATS-BENCH-SSS---------------------------------------#
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_loc', default='./dataset/cifardata/', type=str, help='dataset folder')
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--init', default='', type=str)
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--presample', default=20, type=int, help='pre-sample size')
#----------------------------------------------------NAS-BENCH-101-------------------------------------------------------#
parser.add_argument('--max_nodes', default=5, type=int, help='maximum number of internal nodes')
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
if args.seed is not None:
    torch.manual_seed(args.seed)


from score_function.score import score
from searchspace import searchspace
import datasets.data as data
from datasets.perturbation_data.perturbation_data import *

if args.ptype=='nasbench101':
    from searchspace.nas_101_encoding import BACKBONE as ENCODING
elif args.ptype=='nasbench201':
    from searchspace.nas_201_encoding import BACKBONE as ENCODING
elif args.ptype=='natsbenchsss':
    from searchspace.nats_sss_encoding import ENCODING
encoding = ENCODING() 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

train_loader = data.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
algo_module = importlib.import_module(f'{args.algo}.{args.algo}_{args.ptype}')
init = getattr(algo_module,args.algo)
ss = getattr(searchspace,args.ptype.upper())
ss = ss(args.dataset,args)

if re.search('se[a-z]*',args.algo):
    par = {'iters':args.iters, 'n':args.n, 'h':args.h, 'w':args.w, 'sl':args.sl, 'atom':args.atom, 'max_evaluations':args.max_evaluations}
elif re.search('rs[a-z]*',args.algo):
    par = {'sample':args.max_evaluations}
else:
    raise("No such algo!")

    
if args.ptype=='nasbench201':
    get_acc = lambda id:ss.get_acc_by_code(encoding.parse_code(id),args)
    get_acc_all = lambda id:ss.get_acc_by_code_all(encoding.parse_code(id),args)
    get_acc_proxy = lambda id:ss.get_acc_by_code(encoding.parse_code(id),args,hp=args.hp)
    get_time_proxy = lambda id:ss.get_training_time_by_code(encoding.parse_code(id),args,hp=args.hp)
    get_net = lambda id:ss.get_net_by_code(encoding.parse_code(id),args)
elif args.ptype=='nasbench101':
    get_acc = lambda id:ss.get_acc_by_code_backbone(encoding.parse_code(*id),args)
    get_acc_proxy = lambda id:ss.get_acc_by_code_backbone(encoding.parse_code(*id),args,hp=args.hp)
    get_time_proxy = lambda id:ss.get_training_time_by_code_backbone(encoding.parse_code(*id),args,hp=args.hp)
    get_net = lambda id:ss.get_net_by_code_backbone(encoding.parse_code(*id),args)
elif args.ptype=='natsbenchsss':
    get_acc = lambda id:ss.get_acc(encoding.parse_code(id),args)
    get_acc_all = lambda id:ss.get_acc_all(encoding.parse_code(id),args)
    get_acc_proxy = lambda id:ss.get_acc(encoding.parse_code(id),args,hp=args.hp)
    get_time_proxy = lambda id:ss.get_training_time(encoding.parse_code(id),args,hp=args.hp)
    get_net = lambda id:ss.get_net(encoding.parse_code(id),args)



if args.ptype=='nasbench101':
    par['max_nodes'] = args.max_nodes

if args.dataset == 'cifar10':
    args.acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    args.acc_type = 'x-test'
    val_acc_type = 'x-valid'

hist_code = []
hist_gbest = []
hist_runtime = []
hist_trainingtime = []
hist_acc = []
hist_valid = []
hist_acc_cifar10 = []
hist_valid_cifar10 = []
hist_acc_cifar100 = []
hist_valid_cifar100 = []
hist_acc_imagenet = []
hist_valid_imagenet = []

print("Problem type: {}".format(args.ptype))
print("Algorithm: {}".format(args.algo))
print("Number of evaluations for each run: {}".format(args.max_evaluations))
print("Sequence length: {}".format(args.sl))
print("Base (choices): {}".format(args.atom))

if args.algo=="se":
    print("Number of searchers: {}".format(args.n))
    print("Number of regions: {}".format(args.h))
    print("Number of possible goods: {}".format(args.w))

if args.algo=="rs":
    par['ss'] = ss

for r in range(args.runs):
    start = time.time()
    g = GaussianNoise(train_loader,device,sigma=args.sigma)
    data, target, noise = g.get_noise_data()
    ff = lambda code:score(code,get_net,train_loader,device,args,data,target,noise)


    # avoid repeatedly scoring same networks
    dictionary = {}
    par['dictionary'] = dictionary
    def dff(code):
        if args.ptype=='nasbench101': #NasBench101 has multiple codes
            index = tuple(np.concatenate([x for x in code]))
        else:
            index = tuple(code)
        if index in dictionary:
            return dictionary[index]
        else:
            r = ff(code)
            dictionary[index] = r
            return r
    par['ff'] = dff
    
    # Initialize search algorithm
    algo = init(**par)
    
    gbest_code,gbest = algo.Search()
    end = time.time()
    training_time=0 # For non-training-free only

    hist_code.append(gbest_code)
    hist_gbest.append(gbest)
    hist_runtime.append(end-start)
    hist_trainingtime.append(training_time)

    if args.ptype=='natsbenchsss' or args.ptype=='nasbench201':
        acc_cifar10,valid_cifar10,acc_cifar100,valid_cifar100,acc_imagenet,valid_imagenet = get_acc_all(gbest_code)
        print("Run {:3d}: gbest = {:.3f};acc1 = {:.2f};valid1 = {:.2f};acc2 = {:.2f};valid2 = {:.2f};acc3 = {:.2f};valid3 = {:.2f}; run time = {:.2f}s;training time = {:.2f}; code = {}".format(r,gbest,acc_cifar10,valid_cifar10,acc_cifar100,valid_cifar100,acc_imagenet,valid_imagenet,end-start,training_time,gbest_code))
        df = pd.DataFrame([[r,gbest,acc_cifar10,valid_cifar10,acc_cifar100,valid_cifar100,acc_imagenet,valid_imagenet,end-start,training_time,gbest_code]],columns=['Run','gbest','cifar10_acc','cifar10_valid','cifar100_acc','cifar100_valid','imagenet_acc','imagenet_valid','run time','training time','code'])
    
        hist_acc_cifar10.append(acc_cifar10)
        hist_valid_cifar10.append(valid_cifar10)
        hist_acc_cifar100.append(acc_cifar100)
        hist_valid_cifar100.append(valid_cifar100)
        hist_acc_imagenet.append(acc_imagenet)
        hist_valid_imagenet.append(valid_imagenet)
        print("Average over {} runs: gbest = {:.3f};acc1 = {:.2f};valid1 = {:.2f};acc2 = {:.2f};valid2 = {:.2f};acc3 = {:.2f};valid3 = {:.2f}; run time = {:.2f}s; training time = {:.2f}s".format(r,mean(hist_gbest),mean(hist_acc_cifar10),mean(hist_valid_cifar10),mean(hist_acc_cifar100),mean(hist_valid_cifar100),mean(hist_acc_imagenet),mean(hist_valid_imagenet),mean(hist_runtime),mean(hist_trainingtime)))
    elif args.ptype=='nasbench101':
        acc,valid = get_acc(gbest_code)
        acc*=100; valid*=100
        print("Run {:3d}: gbest = {:.3f};acc = {:.2f};valid = {:.2f}; run time = {:.2f}s;training time = {:.2f}; code = {}".format(r,gbest,acc,valid,end-start,training_time,gbest_code))
        df = pd.DataFrame([[r,gbest,acc,valid,end-start,training_time,gbest_code]],columns=['Run','gbest','acc','valid','run time','training time','code'])

        hist_acc.append(acc)
        hist_valid.append(valid)
        print("Average over {} runs: gbest = {:.3f};acc = {:.2f};valid = {:.2f}; run time = {:.2f}s; training time = {:.2f}s".format(r,mean(hist_gbest),mean(hist_acc),mean(hist_valid),mean(hist_runtime),mean(hist_trainingtime)))

          
print("Average gbest over {} runs: {}".format(args.runs,mean(hist_gbest)))
print("Average run time over {} runs: {:.2f}s".format(args.runs,mean(hist_runtime)))
print("Maximum gbest over {} runs: {}".format(args.runs,max(hist_gbest)))
print("Minimum gbest over {} runs: {}".format(args.runs,min(hist_gbest)))
if args.ptype=='natsbenchsss' or args.ptype=='nasbench201':
    print("Average acc over {} runs: {:.2f}".format(args.runs,mean(hist_acc_cifar10)))
    print("Average valid over {} runs: {:.2f}".format(args.runs,mean(hist_valid_cifar10)))
    print("Average acc over {} runs: {:.2f}".format(args.runs,mean(hist_acc_cifar100)))
    print("Average valid over {} runs: {:.2f}".format(args.runs,mean(hist_valid_cifar100)))
    print("Average acc over {} runs: {:.2f}".format(args.runs,mean(hist_acc_imagenet)))
    print("Average valid over {} runs: {:.2f}".format(args.runs,mean(hist_valid_imagenet)))
elif args.ptype=='nasbench101':
    print("Average acc over {} runs: {:.2f}".format(args.runs,mean(hist_acc)))
    print("Average valid over {} runs: {:.2f}".format(args.runs,mean(hist_valid)))