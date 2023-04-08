import torch
import numpy as np
from utils import add_dropout, init_network 

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()

_scores = {
        'hook_logdet': hooklogdet,
        'random': random_score
        }

def get_score_func(score_name):
    return _scores[score_name]

def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()

def score(network,pack,device,args):
    network = network.cuda()
    s=[]
    try:
        if args.dropout:
            add_dropout(network, args.sigma)
        if args.init != '':
            init_network(network, args.init)
        if 'hook_' in args.score:
            network.K = []
            network.rn = 0
            network.n_conv=0
            network.channel = 0
            def counting_forward_hook(module, inp, out):
                try:
                    if not module.visited_backwards:
                        return
                    if isinstance(inp, tuple):
                        inp = inp[0]

                    arr = inp.detach().cpu().numpy()
                    network.K.append(arr)
                    
                    network.rn = network.rn + 1
                    network.channel+=arr.shape[1]
                except Exception as e:
                    pass

                
            def counting_backward_hook(module, inp, out):
                module.visited_backwards = True

            def counting_forward_hook_conv(module, inp, out):
                try:
                    if not module.visited_backwards_conv:
                        return
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    arr = inp.detach().cpu().numpy()               
                    network.n_conv = network.n_conv + 1
                    network.channel+=arr.shape[1]
                except Exception as e:
                    pass

                
            def counting_backward_hook_conv(module, inp, out):
                module.visited_backwards_conv = True
                
                
            for name, module in network.named_modules():
                if 'Pool' in str(type(module)):
                    module.register_forward_hook(counting_forward_hook)
                    module.register_backward_hook(counting_backward_hook)
                
                if 'Conv' in str(type(module)):
                    module.register_forward_hook(counting_forward_hook_conv)
                    module.register_backward_hook(counting_backward_hook_conv)
                    
        
        for j in range(args.maxofn):
            if  len(pack) == 2:
                x, target = pack
            elif len(pack) == 3:
                x, target, noise = pack
                x = x + noise
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)
            
    
            if 'hook_' in args.score:
                network(x2.to(device))
                rn = network.rn
                n_conv = network.n_conv
                metric = network.K
                channel = network.channel
            else:
                s.append(get_score_func(args.score)(jacobs, labels))
        if args.ptype == 'nasbench101':
            return metric,n_conv,channel
        elif args.ptype == 'nasbench201':
            return metric,n_conv,channel
        else:
            return metric,n_conv,channel
    except Exception as e:
        return np.nan