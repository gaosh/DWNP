import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from models.resnet_hyper import ResNet
from models.gate_function import virtual_gate
from models.hypernet import Simplified_Gate, HyperStructure, DynamicEmbedding

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the vgg')
parser.add_argument('--world_size', type=int, default=10,
                    help='depth of the vgg')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--method', type=str, default='dynamic')
#_world_size_10_local_steps_5_hyper_steps_1_hyper_inte_10_reg_w_2_hn_hetero-dir
#_world_size_10_local_steps_5_hyper_steps_1_hyper_inte_10_reg_w_2_hn_hetero-dir-pruned-dynamic
parser.add_argument('--extra_str', default='_world_size_10_local_steps_5_hyper_steps_1_hyper_inte_10_reg_w_2_hn_hetero-dir', type=str)
parser.add_argument('--hn_arch', default='hn', type=str)

dir = '/datasets/cifar10/'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

depth = args.depth

if not os.path.exists(args.save):
    os.makedirs(args.save)

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = ResNet(depth=depth, gate_flag=True, norm_layer=nn.GroupNorm)
if args.cuda:
    model.cuda()
model_name = 'resnet'
if args.method == 'dynamic':
    stat_dict = torch.load('./checkpoint/%s-pruned-dynamic.pth.tar' % (model_name + str(depth) + args.extra_str))
elif args.method == 'fl1':
    stat_dict = torch.load('./checkpoint/%s_fl1_pruned.pth.tar' % (model_name + str(depth)))
else:
    stat_dict = torch.load('./checkpoint/%s-pruned.pth.tar'%(model_name+str(depth)+args.extra_str))

if args.method != 'fl1':
    print(stat_dict.keys())
    model.load_state_dict(stat_dict['net'])
    model.cuda()
    # resnet56-pruned.pt
    width, structure = model.count_structure()
    # resnet56_world_size_10_local_steps_5_hyper_steps_1_hyper_inte_10_reg_w_2_hn_hetero-dir-pruned
    # hyper_net = Simplified_Gate(structure=structure, T=0.4, base=3.0,)
    if args.hn_arch == 'simple':
        hyper_net = Simplified_Gate(structure=structure, T=0.4, base=3.0)
    else:
        hyper_net = HyperStructure(structure=structure, T=0.4, base=3.0)
    # print(net)
    hyper_net.cuda()
    hyper_net.load_state_dict(stat_dict['hyper_net'])

    hyper_net.eval()
    with torch.no_grad():
        vector = hyper_net()
    #     print(vector)
    print(stat_dict['acc'])
    # vector = stat_dict['arch_vector']
    parameters = hyper_net.transfrom_output(vector.detach())
else:
    parameters = stat_dict['vectors']


# hyper_net.train()
# with torch.no_grad():
#     soft_vector = hyper_net()
#     print(vector)
# soft_parameters = hyper_net.transfrom_output(soft_vector.detach())

cfg = []
for i in range(len(parameters)):
    # if int(parameters[i].sum().item()) == 0:
    #     cfg.append(1)
    #     ind = torch.argmax(soft_parameters[i])
    #     parameters[i][ind] = 1
    # else:
        cfg.append(int(parameters[i].sum().item()))

print(cfg)

newmodel = ResNet(depth=depth, cfg=cfg, gate_flag=True, norm_layer=nn.GroupNorm)
newmodel.cuda()

#layer_id_in_cfg = 0
old_modules = list(model.modules())
new_modules = list(newmodel.modules())
start_mask = torch.ones(3)
soft_gate_count = 0
conv_count =0
end_mask = parameters[soft_gate_count]

#norm_layer =  nn.BatchNorm2d
norm_layer = nn.modules.normalization.GroupNorm

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, norm_layer):
        # print(m0)
        # print(m1)
        idx1 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        if layer_id==2:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            if hasattr(m0, 'running_mean'):
                m1.running_mean = m0.running_mean.clone()
            if hasattr(m0, 'running_var'):
                m1.running_var = m0.running_var.clone()
            #print(layer_id)
            continue
        elif isinstance(old_modules[layer_id + 2], virtual_gate):
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.

            print(m0)
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            if hasattr(m0, 'running_mean'):
                m1.running_mean = m0.running_mean.clone()
            if hasattr(m0, 'running_var'):
                m1.running_var = m0.running_var.clone()
            # We need to set the channel selection layer.
            # m2 = new_modules[layer_id + 2]
            # m2.indexes.data.zero_()
            # m2.indexes.data[:] = 1.0

        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            if hasattr(m0, 'running_mean'):
                m1.running_mean = m0.running_mean.clone()
            if hasattr(m0, 'running_var'):
                m1.running_var = m0.running_var.clone()

    elif isinstance(m0, nn.Conv2d):
        #print(old_modules[layer_id+2])
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue

        if isinstance(old_modules[layer_id+3], virtual_gate):
            print(conv_count)
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            print(m1.weight.data.size())

            m0_next = old_modules[layer_id+4]
            m1_next = new_modules[layer_id+4]
            print(m0_next)
            print(m1_next)
            if isinstance(m0_next, nn.Conv2d):

                w1 = m0_next.weight.data[:, idx1.tolist(), :, :].clone()
                m1_next.weight.data = w1.clone()
                print(m1_next.weight.data.size())

            soft_gate_count += 1
            start_mask = end_mask.clone()
            if soft_gate_count < len(parameters):
                end_mask = parameters[soft_gate_count]

            continue
        if isinstance(old_modules[layer_id -1], virtual_gate):
            continue
        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.

        m1.weight.data = m0.weight.data.clone()

    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        #m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

model.cpu()
newmodel.cpu()
t_o=print_model_param_nums(model)
t_n=print_model_param_nums(newmodel)
print_model_param_flops(model, input_res=32)
print_model_param_flops(newmodel, input_res=32)



all_parameters = torch.cat(parameters)
print(all_parameters)
pruning_rate = float((all_parameters==1).sum())/float(all_parameters.size(0))
print(pruning_rate)

model_new = ResNet(depth=depth, gate_flag=True, cfg=cfg, norm_layer=nn.GroupNorm)

newmodel_ng_ms = list(model_new.modules())
newmodel_ms = list(newmodel.modules())
#model_new.set_training_flag(False)
# print(resnet_50_ms)
#print(mymbnet_v2_ms)
for m in newmodel_ms:
    if isinstance(m, virtual_gate):
        newmodel_ms.remove(m)

for m in newmodel_ng_ms:
    if isinstance(m, virtual_gate):
        newmodel_ng_ms.remove(m)

if args.method == 'dynamic':
    dynamic_emb = DynamicEmbedding(structure=structure, T=0.4, base=3.0, num_clients=args.world_size)
    dynamic_emb.cuda()
    dynamic_emb.load_state_dict(stat_dict['dynamic_emb'])
    dynamic_emb.eval()
    dynamic_cfg = {}
    for i in range(args.world_size):
        with torch.no_grad():
            task_id = torch.Tensor([i]).long().cuda()
            client_vector = dynamic_emb(task_id)
            vector_index = vector.nonzero()

            dynamic_cfg[i] = client_vector[vector_index].cpu()

print(len(newmodel_ms))
print(len(newmodel_ng_ms))



#print(mbnet_v2)
for layer_id in range(len(newmodel_ng_ms)):
    m0 = newmodel_ms[layer_id]#         newmodel_ng_ms.remove(m)

    m1 = newmodel_ng_ms[layer_id]
    if isinstance(m0, norm_layer):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        if hasattr(m0, 'running_mean'):
            m1.running_mean = m0.running_mean.clone()
        if hasattr(m0, 'running_var'):
            m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Conv2d):
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

if args.method == 'dynamic':
    torch.save({'cfg': cfg, 'state_dict': model_new.state_dict(), 'dynamic_cfg': dynamic_cfg,
                'pruned_index':vector_index,  'dynamic_emb':stat_dict['dynamic_emb']},
               os.path.join(args.save, './checkpoint/%s_new.pth.tar' % (model_name + str(depth))))
else:
    torch.save({'cfg': cfg, 'state_dict': model_new.state_dict()}, os.path.join(args.save, './checkpoint/%s_new.pth.tar'%(model_name+str(depth))))






