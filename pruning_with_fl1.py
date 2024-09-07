from train import *
from utils import *
from repeat_dataloader import MultiEpochsDataLoader
#from models.resnet_gate import ResNet as ResNet_gate
from models.resnet_hyper import ResNet as ResNet_hyper
from models.hypernet import Simplified_Gate, Simple_PN, HyperStructure, DynamicEmbedding, transfrom_output
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
    model_name = 'resnet'

    prefix = '_world_size_' + str(args.world_size) + '_local_steps_' + str(args.local_steps) + '_hyper_steps_' + str(args.hyper_steps) +\
                 '_hyper_inte_' + str(args.hyper_interval) + '_reg_w_' + str(args.reg_w) + '_' + args.hn_arch + '_' + args.partition
    logger = Logger('results', prefix)

    if rank == 0: print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cls_per_client = 10 // size
    assert cls_per_client > 0

    class_idx = set(np.arange(cls_per_client*rank,cls_per_client*(rank+1)))

    trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=True, transform=transform_train)
    dataset_size = len(trainset.targets)
    args.num_iter = dataset_size // args.batch_size
    if args.split_test:
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True,
                                               transform=transform_test)

    if rank == 0 and not args.split_test:
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True, transform=transform_test)
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
        lengths=[len(trainset) - int(0.1 * len(trainset)), int(0.1 * len(trainset))]
    )

    trainloader = MultiEpochsDataLoader(trainset, batch_size=int(args.batch_size / size), num_workers=1, shuffle=True,
                                        pin_memory=False)
    validloader = MultiEpochsDataLoader(valset, batch_size=int(args.batch_size / size), num_workers=1, pin_memory=False)
    if args.split_test:
        testloader = torch.utils.data.DataLoader(testset, batch_size=int(200 / size), shuffle=False, num_workers=1,
                                                 pin_memory=False)

    net = ResNet_hyper(depth=depth, gate_flag=True, norm_layer=nn.GroupNorm)
    width, structure = net.count_structure()


    net.cuda()

    model_name = 'resnet'
    if args.method == 'dynamic':
        stat_dict = torch.load('./checkpoint/%s-pruned-dynamic.pth.tar' % (model_name + str(depth) + args.extra_str))
    else:
        stat_dict = torch.load('./checkpoint/%s-pruned.pth.tar' % (model_name + str(depth) + args.extra_str))
    net.load_state_dict(stat_dict['net'])
    net.enable_fn()

    best_acc = 0
    if rank == 0:
        distributed_vaild(0, net, trainloader, best_acc, hyper_net=None,
                                     dynamic_emb=None,
                                     model_string=None,
                                     stage='valid_model',
                                     logger=logger, args=args)

    else:
        distributed_vaild(0, net, trainloader, best_acc, hyper_net=None,
                          dynamic_emb=None,
                          model_string=None,
                          stage='valid_model',
                          logger=logger, args=args)
    fn_list = net.final_fn()
    threshold = 0.05
    for i in range(len(fn_list)):
        dist.all_reduce(fn_list[i])
    if rank == 0:
        net.cpu()
        vector_list = []
        for i in range(len(fn_list)):
            vector_list.append(torch.ones(fn_list[i].size(0)))
        vectors = torch.cat(vector_list, dim=0)
        feature_importance = torch.cat(fn_list, dim=0).cpu()

        size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnet(net)
        resource_reg = Flops_constraint_resnet(args.p, size_kernel, size_out, size_group, size_inchannel,
                                               size_outchannel,
                                               w=args.reg_w, HN=True, structure=structure, )

        end_index = vectors.size(0) - 1
        start_index = vectors.size(0) - 1 - args.prune_size
        _, index = torch.sort(feature_importance, descending=True)

        while resource_reg(vectors) > threshold:

            print(resource_reg(vectors))
            vectors = vectors[index]
            vectors[start_index:end_index] = 0


            vectors = vectors.gather(0, index.squeeze().argsort())

            end_index = start_index
            start_index = start_index-args.prune_size

        vectors = transfrom_output(vectors, structure)

        cfg = []
        for i in range(len(vectors)):
            if int(vectors[i].sum().item()) == 0:
                max_idx = fn_list[i].cpu().argmax()
                vectors[i][max_idx] = 1

            cfg.append(int(vectors[i].sum().item()))

        print(cfg)

        torch.save({ 'state_dict': net.state_dict(),'vectors':vectors},
                   './checkpoint/%s_fl1_pruned.pth.tar' % (model_name + str(depth)))

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

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--partition', default='hetero-dir',choices=['hetero-dir', 'class', 'homo'])
parser.add_argument('--method', default='static',choices=['static', 'dynamic'])
parser.add_argument('--split_test', default=True, type=str2bool)
parser.add_argument('--extra_str', default='_world_size_10_local_steps_5_hyper_steps_1_hyper_inte_10_reg_w_2_hn_hetero-dir', type=str)
parser.add_argument('--prune_size', default=10, type=float)
args = parser.parse_args()

size = args.world_size
processes = []

dir = '/datasets/cifar10/'
# args.cuda = not args.no_cuda and torch.cuda.is_available()

for rank in range(size):
    p = Process(target=init_processes, args=(rank, size, args, run))
    p.start()
    processes.append(p)

for p in processes:
    p.join()