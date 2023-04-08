#NasBench201
from .nas_201_api import NASBench201API as API201
from .models import get_cell_based_tiny_net
#NasBench101
from .nasbench import api as API101
from nas_101_api.model import Network
from nas_101_api.model_spec import ModelSpec
#NATSBench-sss
from nats_api import create as create_sss

import pandas as pd
import itertools
import numpy as np

class NASBENCH201:
    '''201'''
    def __init__(self, dataset,args):
        self.dataset = dataset
        print("Loading api...")
        self.api = API201('./searchspace/NAS-Bench-201-v1_1-096897.pth',verbose=False)
        print("Finished loading.")
        self.operations = ['none',
                                            'skip_connect',
                                            'nor_conv_1x1', 
                                            'nor_conv_3x3', 
                                            'avg_pool_3x3' ]
        self.args=args
    def __len__(self):
        return 15625

    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_net(uid)
            yield uid, network

    def __getitem__(self, index):
        return index

    def get_net(self,index):
        index = self.api.query_index_by_arch(index)
        if self.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = self.dataset
        config = self.api.get_net_config(index, dataname)
        config['num_classes'] = 1
        network = get_cell_based_tiny_net(config)
        return network
    
    def get_acc_all(self,index,args):
        index = self.api.query_index_by_arch(index)
        information = self.api.arch2infos_dict[index]['200']

        valid_info_cifar10 = information.get_metrics('cifar10-valid', 'x-valid')
        valid_acc_cifar10 = valid_info_cifar10['accuracy']
        test__info_cifar10 = information.get_metrics('cifar10', 'ori-test')
        test_acc_cifar10 = test__info_cifar10['accuracy']

        valid_info_cifar100 = information.get_metrics('cifar100', 'x-valid')
        test__info_cifar100 = information.get_metrics('cifar100', 'x-test')
        valid_acc_cifar100 = valid_info_cifar100['accuracy']
        test_acc_cifar100 = test__info_cifar100['accuracy']

        valid_info_imagenet = information.get_metrics('ImageNet16-120', 'x-valid')
        test__info_imagenet = information.get_metrics('ImageNet16-120', 'x-test')
        valid_acc_imagenet = valid_info_imagenet['accuracy']
        test_acc_imagenet = test__info_imagenet['accuracy']

        return test_acc_cifar10,valid_acc_cifar10,test_acc_cifar100,valid_acc_cifar100,test_acc_imagenet,valid_acc_imagenet

    def get_acc(self, index, args, hp='200'):
        index = self.api.query_index_by_arch(index)
        information = self.api.arch2infos_dict[index][hp]
        if args.dataset == 'cifar10':
            valid_info = information.get_metrics('cifar10-valid', 'x-valid')
            valid_acc = valid_info['accuracy']
            test__info = information.get_metrics('cifar10', 'ori-test')
            test_acc = test__info['accuracy']
        else:
            valid_info = information.get_metrics(args.dataset, 'x-valid')
            test__info = information.get_metrics(args.dataset, 'x-test')
            valid_acc = valid_info['accuracy']
            test_acc = test__info['accuracy']
        return test_acc,valid_acc

    def get_acc_by_code(self,code,args,hp='200'):
        if hp is not str:
            hp = str(hp)
        index = self.get_index_by_code(code,args)
        information = self.api.arch2infos_dict[index][hp]
        if args.dataset == 'cifar10':
            valid_info = information.get_metrics('cifar10-valid', 'x-valid')
            valid_acc = valid_info['accuracy']
            test__info = information.get_metrics('cifar10', 'ori-test')
            test_acc = test__info['accuracy']
        else:
            valid_info = information.get_metrics(args.dataset, 'x-valid')
            test__info = information.get_metrics(args.dataset, 'x-test')
            valid_acc = valid_info['accuracy']
            test_acc = test__info['accuracy']
        return test_acc,valid_acc

    def get_acc_by_code_all(self,code,args):
        index = self.get_index_by_code(code,args)
        information = self.api.arch2infos_dict[index]['200']

        valid_info_cifar10 = information.get_metrics('cifar10-valid', 'x-valid')
        valid_acc_cifar10 = valid_info_cifar10['accuracy']
        test__info_cifar10 = information.get_metrics('cifar10', 'ori-test')
        test_acc_cifar10 = test__info_cifar10['accuracy']

        valid_info_cifar100 = information.get_metrics('cifar100', 'x-valid')
        test__info_cifar100 = information.get_metrics('cifar100', 'x-test')
        valid_acc_cifar100 = valid_info_cifar100['accuracy']
        test_acc_cifar100 = test__info_cifar100['accuracy']

        valid_info_imagenet = information.get_metrics('ImageNet16-120', 'x-valid')
        test__info_imagenet = information.get_metrics('ImageNet16-120', 'x-test')
        valid_acc_imagenet = valid_info_imagenet['accuracy']
        test_acc_imagenet = test__info_imagenet['accuracy']

        return test_acc_cifar10,valid_acc_cifar10,test_acc_cifar100,valid_acc_cifar100,test_acc_imagenet,valid_acc_imagenet

    def get_index_by_code(self,code,args):
        node_str = ""
        base=-0
        for j in range(1,4):
            node_str += '|'
            for k in range(0,j):
                node_str = node_str + self.operations[int(code[base])] + '~'+ str(k) +'|'
                base+=1
            node_str += '+'
        node_str = node_str[0:-1]
        index = self.api.query_index_by_arch(node_str)
        return index
    
    def get_net_by_code(self,code,args):
        index = self.get_index_by_code(code,args)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset

        config = self.api.get_net_config(index, dataname)
        config['num_classes'] = 1
        network = get_cell_based_tiny_net(config)
        return network
    
    def get_training_time_by_code(self,code,args,hp):
        index = self.get_index_by_code(code,args)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset
        info = self.api.get_more_info(
            index, dataname, iepoch=None, hp=hp, is_random=True
        )
        time_cost = info["train-all-time"] + info["valid-per-time"]
        
        return time_cost

    def get_training_time(self,index,args,hp):
        index = self.api.query_index_by_arch(index)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset
        info = self.api.get_more_info(
            index, dataname, iepoch=None, hp=hp, is_random=True
        )
        time_cost = info["train-all-time"] + info["valid-per-time"]
        
        return time_cost



class NASBENCH101:
    '''101'''
    def __init__(self, dataset,args):
        self.dataset = dataset
        print("Loading api...") 
        #self.api = API101.NASBench('./searchspace/nasbench_full.tfrecord')
        self.api = API101.NASBench('./searchspace/nasbench_only108.tfrecord')
        print("Finished loading.")
        self.args=args

    def __len__(self):
        return len(self.api.hash_iterator())
    
    def __iter__(self):
        for unique_hash in self.api.hash_iterator():
            network,g = self.get_net(unique_hash)
            yield unique_hash, network,g
    
    def __getitem__(self, index):
        return next(itertools.islice(self.api.hash_iterator(), index, None))
    
    def get_net(self, unique_hash):
        spec = self.get_spec(unique_hash)
        g = spec.visualize()
        network = Network(spec, self.args)
        return network,g

    def get_acc(self, index, args,hp, trainval=True):
        spec = self.get_spec(index)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = 0.
        maxval = 0.
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap['final_test_accuracy']
                newval = statmap['final_validation_accuracy']
                if newacc > maxacc:
                    maxacc = newacc
                    maxval = newval
        return maxacc,maxval

    def get_final_accuracy(self, uid, acc_type, trainval):
        return self.get_acc(uid, trainval)
    
    def get_acc_by_matrix(self,matrix,ops,args):
        model_spec = ModelSpec(matrix,ops)
        data = self.api.query(model_spec)
        return data['test_accuracy']

    def get_acc_by_code(self,code,args,hp=108):
        c1 = code[0]
        c2 = code[1]
        matrix = [[0, 0, 0, 0, 0, 0, 0],    # 6 input layer 
                            [0, 0, 0, 0, 0, 0, 0],    # 5
                            [0, 0, 0, 0, 0, 0, 0],    # 4
                            [0, 0, 0, 0, 0, 0, 0],    # 3
                            [0, 0, 0, 0, 0, 0, 0],    # 2
                            [0, 0, 0, 0, 0, 0, 0],    # 1
                            [0, 0, 0, 0, 0, 0, 0]]   # output layer
        counter = 0
        for i in range(6,0,-1):
            for j in range(1,i+1):
                matrix[6-i][6-i+j] = c1[counter]
                counter+=1
        ops = ['input', '' ,'' ,'' ,'' ,'' , 'output']
        for i,x in enumerate(c2):
            if x==0:
                ops[i+1] = 'conv1x1-bn-relu'
            elif x==1:
                ops[i+1] = 'conv3x3-bn-relu'
            elif x==2:
                ops[i+1] = 'maxpool3x3'
        model_spec = ModelSpec(matrix,ops)
        
        try:
            data = self.api.query(model_spec,hp)
            return data['test_accuracy'],data['validation_accuracy']
        except Exception as e:
            return 0,0

    def get_acc_by_code_backbone(self,code,args,hp=108):
        matrix = code[0]
        ops = code[1]
        model_spec = ModelSpec(matrix,ops)
        try:
            data = self.api.query(model_spec,hp)
            return data['test_accuracy'],data['validation_accuracy']
        except Exception as e:
            print(e)
            return 0,0

    def get_training_time_by_code_backbone(self,code,args,hp=108):
        matrix = code[0]
        ops = code[1]
        model_spec = ModelSpec(matrix,ops)
        try:
            data = self.api.query(model_spec,hp)
            return data['training_time']
        except Exception as e:
            print(e)
            return 0
    def get_training_time_by_code(self,code,args,hp=108):
        c1 = code[0]
        c2 = code[1]
        matrix = [[0, 0, 0, 0, 0, 0, 0],    # 6 input layer 
                            [0, 0, 0, 0, 0, 0, 0],    # 5
                            [0, 0, 0, 0, 0, 0, 0],    # 4
                            [0, 0, 0, 0, 0, 0, 0],    # 3
                            [0, 0, 0, 0, 0, 0, 0],    # 2
                            [0, 0, 0, 0, 0, 0, 0],    # 1
                            [0, 0, 0, 0, 0, 0, 0]]   # output layer
        counter = 0
        for i in range(6,0,-1):
            for j in range(1,i+1):
                matrix[6-i][6-i+j] = c1[counter]
                counter+=1
        ops = ['input', '' ,'' ,'' ,'' ,'' , 'output']
        for i,x in enumerate(c2):
            if x==0:
                ops[i+1] = 'conv1x1-bn-relu'
            elif x==1:
                ops[i+1] = 'conv3x3-bn-relu'
            elif x==2:
                ops[i+1] = 'maxpool3x3'
        model_spec = ModelSpec(matrix,ops)
        try:
            data = self.api.query(model_spec,hp)
            return data['training_time']
        except Exception as e:
            return 1e8

    def get_net_by_code(self, code, args):
        c1 = code[0]
        c2 = code[1]
        matrix = [[0, 0, 0, 0, 0, 0, 0],    # 6 input layer 
                            [0, 0, 0, 0, 0, 0, 0],    # 5
                            [0, 0, 0, 0, 0, 0, 0],    # 4
                            [0, 0, 0, 0, 0, 0, 0],    # 3
                            [0, 0, 0, 0, 0, 0, 0],    # 2
                            [0, 0, 0, 0, 0, 0, 0],    # 1
                            [0, 0, 0, 0, 0, 0, 0]]   # output layer
        counter = 0
        for i in range(6,0,-1):
            for j in range(1,i+1):
                matrix[6-i][6-i+j] = c1[counter]
                counter+=1
        ops = ['input', '' ,'' ,'' ,'' ,'' , 'output']
        for i,x in enumerate(c2):
            if x==0:
                ops[i+1] = 'conv1x1-bn-relu'
            elif x==1:
                ops[i+1] = 'conv3x3-bn-relu'
            elif x==2:
                ops[i+1] = 'maxpool3x3'
        model_spec = ModelSpec(matrix,ops)
        network = Network(model_spec, args)
        return network

    def get_net_by_code_backbone(self, code, args):
        matrix = code[0]
        ops = code[1]
        try:
            model_spec = ModelSpec(matrix,ops)
            network = Network(model_spec, args)
            return network
        except Exception as e:
            print(e)
            return None

    def get_spec(self, unique_hash):
        matrix = self.api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self.api.fixed_statistics[unique_hash]['module_operations']
        spec = ModelSpec(matrix, operations)
        return spec


class NATSBENCHSSS:
    '''sss'''
    def __init__(self, dataset,args):
        self.dataset = dataset
        print("Loading api...")
        self.api = create_sss("./searchspace/NATS-sss-v1_0-50262-simple/", 'sss', fast_mode=True, verbose=False)
        print("Finished loading.")
        self.args=args

    def __len__(self):
        return 32768

    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_net(uid)
            yield uid, network

    def __getitem__(self, index):
        return index

    def get_net(self,index,args):
        index = self.api.query_index_by_arch(index)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset
        config = self.api.get_net_config(index, dataname)
        config['num_classes'] = 1
        network = get_cell_based_tiny_net(config)
        return network
    
    def get_acc(self, index, args,hp=90):
        index = self.api.query_index_by_arch(index)
        info = self.api.get_more_info(index, args.dataset, hp= hp)
        validation_accuracy, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset=args.dataset, hp=hp)
        return info['test-accuracy'],validation_accuracy

    def get_acc_all(self, index, args):
        info_cifar10 = self.api.get_more_info(index, 'cifar10', hp=90)
        info_cifar100 = self.api.get_more_info(index, 'cifar100', hp=90)
        info_imagenet = self.api.get_more_info(index, 'ImageNet16-120', hp=90)

        validation_accuracy_cifar10, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='cifar10', hp=90)
        validation_accuracy_cifar100, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='cifar100', hp=90)
        validation_accuracy_imagenet, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='ImageNet16-120', hp=90)        
        return info_cifar10['test-accuracy'],validation_accuracy_cifar10,info_cifar100['test-accuracy'],validation_accuracy_cifar100,info_imagenet['test-accuracy'],validation_accuracy_imagenet

    def get_training_time(self,index,args,hp):
        validation_accuracy, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset=args.dataset, hp=hp)
        return time_cost
