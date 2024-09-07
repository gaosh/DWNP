from train import *
from utils import *
from repeat_dataloader import MultiEpochsDataLoader
#from models.resnet_gate import ResNet as ResNet_gate
from models.resnet_hyper import ResNet as ResNet_hyper
from models.mobilenetv2_hyper import MobileNetV2
from models.hypernet import Simplified_Gate, Simple_PN, HyperStructure, DynamicEmbedding
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from optimizer import AdamW
from torch.utils.data.dataset import random_split
from data_util import partition_data

from torch.multiprocessing import Process
import torch.distributed as dist
import numpy as np
from utils import Logger

def init_processes(rank, size, args, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(np.random.randint(10000, 30000))
    gpus = args.gpu_visible.split(',')
    num_gpus = len(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[rank % num_gpus]
    # os.environ["CUDA_VISIBLE_DEVICES"] = '5' if rank == 0 else '6'

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run(args):

    torch.set_num_threads(1)

    rank = dist.get_rank()
    size = dist.get_world_size()

    depth = args.depth
    model_name = args.model_name

    prefix = '_world_size_' + str(args.world_size) + '_local_steps_' + str(args.local_steps) + '_hyper_steps_' + str(args.hyper_steps) +\
                 '_hyper_inte_' + str(args.hyper_interval) + '_reg_w_' + str(args.reg_w) + '_' + args.hn_arch + '_' + args.partition
    logger = Logger('results', prefix)

    if rank == 0: print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #
        # transforms.RandomRotation(10),  # Rotates the image to a specified angel
        # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Performs actions like zooms, change shear angles.
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cls_per_client = 10 // size
    assert cls_per_client > 0

    class_idx = set(np.arange(cls_per_client*rank,cls_per_client*(rank+1)))

    trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=False, transform=transform_train)
    dataset_size = len(trainset.targets)
    args.num_iter = dataset_size // args.batch_size
    if args.split_test:
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=False,
                                               transform=transform_test)

    # idx = [tgt in class_idx for tgt in trainset.targets]
    # idx = np.arange(len(trainset.targets))[idx]
    # trainset.targets = [trainset.targets[id] for id in idx]
    # trainset.data = [trainset.data[id] for id in idx]

    if rank == 0 and not args.split_test:
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1, pin_memory=False)

    if args.partition == 'class':
        idx = [tgt in class_idx for tgt in trainset.targets]
        idx = np.arange(len(trainset.targets))[idx]
        print(idx[:10])
        trainset.targets = [trainset.targets[id] for id in idx]
        trainset.data = [trainset.data[id] for id in idx]
    elif args.partition == 'hetero-dir':
        # if rank==0:
        if args.split_test:
            net_dataidx_map, net_dataidx_map_test = partition_data(trainset, args=args, test_dataset=testset)
            trainset.targets = [trainset.targets[id] for id in net_dataidx_map[rank]]
            trainset.data = [trainset.data[id] for id in net_dataidx_map[rank]]

            testset.targets = [testset.targets[id] for id in net_dataidx_map_test[rank]]
            testset.data = [testset.data[id] for id in net_dataidx_map_test[rank]]

        else:
            net_dataidx_map = partition_data(trainset, args=args)
            trainset.targets =[trainset.targets[id] for id in net_dataidx_map[rank]]
            trainset.data =[trainset.data[id] for id in net_dataidx_map[rank]]
            print(len(trainset.targets))





    _, valset = random_split(
        trainset,
        lengths=[len(trainset)-int(0.1*len(trainset)), int(0.1*len(trainset))]
    )
    # train_sampler,val_sampler = TrainVal_split(trainset, 0.1, shuffle_dataset=True)

    trainloader = MultiEpochsDataLoader(trainset, batch_size=int(args.batch_size/size), num_workers=1,shuffle=True, pin_memory=False)
    validloader = MultiEpochsDataLoader(valset, batch_size=int(args.batch_size/size), num_workers=1, pin_memory=False)
    if args.split_test:
        testloader = torch.utils.data.DataLoader(testset, batch_size=int(200/size), shuffle=False, num_workers=1,
                                                 pin_memory=False)
    if args.model_name == 'mobnetv2':
        net = MobileNetV2( norm_layer=nn.GroupNorm)
        if rank==0: print(net)
    elif args.mode_name == 'resnet-56':
        net = ResNet_hyper(depth=depth, gate_flag=True, norm_layer=nn.GroupNorm)
    width, structure = net.count_structure()

    if args.hn_arch == 'simple':
        hyper_net = Simplified_Gate(structure=structure, T=0.4, base=args.base)
    else:
        hyper_net = HyperStructure(structure=structure, T=0.4, base=args.base)

    if args.model_name == 'mobnetv2':
        size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_mobnet(net)
        resource_reg = Flops_constraint_mobnet(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                           w=args.reg_w, HN=True, structure=structure, )
    else:
        size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnet(net)
        resource_reg = Flops_constraint_resnet(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                           w=args.reg_w, HN=True, structure=structure, )

    if args.method == 'dynamic':
        dynamic_emb = DynamicEmbedding(structure=structure, T=0.4, base=args.base, num_clients=args.world_size)

        if args.model_name == 'mobnetv2':
            resource_reg_dynamic = Flops_constraint_mobnet(args.d_p, size_kernel, size_out, size_group, size_inchannel,
                                                           size_outchannel,
                                                           w=args.reg_w, HN=True, structure=structure, )
        else:
            resource_reg_dynamic = Flops_constraint_resnet(args.d_p, size_kernel, size_out, size_group, size_inchannel,size_outchannel,
                                                       w=args.reg_w, HN=True, structure=structure, )
        dynamic_emb.cuda()

    Epoch = args.epoch

    hyper_net.cuda()
    net.cuda()

    if args.opt == 'AdamW':
        if args.hn_arch == 'simple':
            hyper_optimizer = AdamW(filter(lambda p: p.requires_grad, hyper_net.parameters()), lr=1e-2,
                                          weight_decay=1e-3)
        else:
            params = list(filter(lambda p: p.requires_grad, hyper_net.parameters()))
            hyper_optimizer = AdamW(params, lr=1e-3, weight_decay=1e-2)

        if args.method == 'dynamic':
            emb_params = list(filter(lambda p: p.requires_grad, dynamic_emb.parameters()))
            emb_optimizer = AdamW(emb_params, lr=1e-3, weight_decay=1e-2)

    elif args.opt == 'Momentum':
        hyper_optimizer = optim.SGD(filter(lambda p: p.requires_grad, hyper_net.parameters()), lr=0.1, momentum=0.9)

    hyper_scheduler = MultiStepLR(hyper_optimizer, milestones=[int(Epoch * 0.5)], gamma=0.1)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                # momentum=0.9)
                                momentum=args.m, weight_decay=args.wd)

    if args.sch == 'first-more':
        scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * Epoch), int(0.75 * Epoch)], gamma=0.1)
    elif args.sch == 'even':
        scheduler = MultiStepLR(optimizer,
                                milestones=[int(1 / 3 * 0.9 * Epoch), int(2 / 3 * 0.9 * Epoch), int(0.9 * Epoch)],
                                gamma=0.1)
    elif args.sch == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(Epoch), eta_min=0)

    if args.warmup:
        base_sch = scheduler
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=base_sch)
        Epoch = Epoch + 5

    best_acc = 0

    train_args = args
    train_args.opt = 'Momentum'

    if args.iter_train:

        if args.method == 'dynamic':
            models = {}
            optimizers = {}
            dataloaders = {}
            models['net'] = net
            models['hyper_net'] = hyper_net
            models['dynamic_emb'] = dynamic_emb
            optimizers['hyper_optimizer'] = hyper_optimizer
            optimizers['emb_optimizer'] = emb_optimizer
            optimizers['net_optimizer'] = optimizer
            dataloaders['train'] = trainloader
            dataloaders['valid'] = validloader
            args.resource_constraint = resource_reg
            args.resource_constraint_dynamic = resource_reg_dynamic
            for epoch in range(0, Epoch):
                # scheduler.step()
                iterative_dynamic_train(epoch, models, optimizers, dataloaders, args=args, logger=logger)
                scheduler.step()
                hyper_scheduler.step()

                if args.split_test:
                    if rank == 0:
                        best_acc = distributed_vaild(epoch, net, testloader, best_acc, hyper_net=hyper_net,
                                                     dynamic_emb=dynamic_emb,
                                                     model_string='%s-cotrain-dynamic' % (model_name),
                                                     stage='valid_model',
                                                     logger=logger, args=args)
                        logger.print_dynamic(epoch)

                    else:
                        distributed_vaild(epoch, net, testloader, best_acc, hyper_net=hyper_net,
                                          dynamic_emb=dynamic_emb,
                                          model_string='%s-cotrain-dynamic' % (model_name),
                                          stage='valid_model',
                                          logger=logger, args=args)
                else:
                    if rank == 0:
                        best_acc = valid(epoch, net, testloader, best_acc, hyper_net=hyper_net, dynamic_emb=dynamic_emb,
                                     model_string='%s-cotrain-dynamic' % (model_name), stage='valid_model',
                                     logger=logger, args=args)

                        logger.print(epoch)

                        if epoch % 5 == 0:
                            logger.save()
        else:

            args.resource_constraint = resource_reg
            for epoch in range(0, Epoch):
                #scheduler.step()
                iterative_train(epoch, net, trainloader, optimizer, hyper_net, validloader, hyper_optimizer, args, logger=logger)
                # retrain(epoch, net, trainloader, optimizer, smooth=args.smooth_flag,alpha=args.alpha, args=train_args, hyper_net=hyper_net)
                scheduler.step()
                hyper_scheduler.step()
                if rank == 0:
                    best_acc = valid(epoch, net, testloader, best_acc, hyper_net=hyper_net,
                                 model_string='%s-cotrain' % (model_name), stage='valid_model', logger=logger, args=args)

                    logger.print(epoch)

                    if epoch % 5 == 0:
                        logger.save()
    else:
        for epoch in range(0, Epoch):
            #scheduler.step()
            retrain(epoch, net, trainloader, optimizer, smooth=args.smooth_flag,alpha=args.alpha, args=train_args, hyper_net=hyper_net)
            scheduler.step()
            if epoch >= args.start_epoch:
                train_hyper(epoch, net, validloader, hyper_optimizer, hyper_net=hyper_net, resource_constraint=resource_reg, args=args)
            hyper_scheduler.step()
            if rank == 0:
                print()
                best_acc = valid(epoch, net, testloader, best_acc, hyper_net=hyper_net,
                             model_string='%s-cotrain' % (model_name), stage='valid_model', args=args)
    # valid(0, net, testloader, best_acc, hyper_net=None, model_string=None, stage='valid_model',)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--stage', default='train-gate', type=str)
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--d_p', default=0.5, type=float)
parser.add_argument('--depth', default=56, type=int)

parser.add_argument('--gpu_visible', default='1', type=str)

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--start_epoch', default=25, type=int)
parser.add_argument('--hyper_interval', default=10, type=int)
parser.add_argument('--hyper_steps', default=1, type=int)
parser.add_argument('--reg_w', default=2, type=float)
parser.add_argument('--base', default=3.0, type=float)

parser.add_argument('--m', default=0.9, type=float)
parser.add_argument('--sch', default='first-more',type=str)
parser.add_argument('--smooth_flag', default=False, type=str2bool)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--warmup', default=False, type=str2bool)

parser.add_argument('--hn_arch', default='hn', type=str)
parser.add_argument('--local_steps', default=1, type=int)
parser.add_argument('--world_size', default=2, type=int)
parser.add_argument('--opt', default='AdamW', choices=['SGD', 'Momentum', 'Adam', 'AdamW'])
parser.add_argument('--iter_train', default=True, type=str2bool)
parser.add_argument('--model_name', default='resnet56',type=str)

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--partition', default='hetero-dir',choices=['hetero-dir', 'class', 'homo'])
parser.add_argument('--method', default='dynamic',choices=['static', 'dynamic'])
parser.add_argument('--split_test', default=True, type=str2bool)
args = parser.parse_args()

size = args.world_size
processes = []

for rank in range(size):
    p = Process(target=init_processes, args=(rank, size, args, run))
    p.start()
    processes.append(p)

for p in processes:
    p.join()