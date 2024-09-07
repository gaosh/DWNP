from train import *
from utils import *
#from models.resnet_gate import ResNet as ResNet_gate
from models.resnet_hyper import ResNet as ResNet_hyper
from models.hypernet import Simplified_Gate, PP_Net, Episodic_mem, Simple_PN, HyperStructure
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from data_util import partition_data
from optimizer import AdamW

from torch.multiprocessing import Process
import torch.distributed as dist

def init_processes(rank, size, args, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '24121'
    gpus = args.gpu_visible.split(',')
    num_gpus = len(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[rank % num_gpus]
    # os.environ["CUDA_VISIBLE_DEVICES"] = '6' if rank > 16 else '3'

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

    rank = dist.get_rank()
    size = dist.get_world_size()

    depth = args.depth
    model_name = 'resnet'
    print('==> Preparing data..')
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

    # if rank == 0:
    #     class_idx = set(np.arange(5))
    # else:
    #     class_idx = set(np.arange(5, 10))

    class_idx = set(np.arange(cls_per_client*rank,cls_per_client*(rank+1)))

    trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=True, transform=transform_train)
    # idx = [tgt in class_idx for tgt in trainset.targets]
    # idx = np.arange(len(trainset.targets))[idx]
    # print(idx[:10])
    # trainset.targets = [trainset.targets[id] for id in idx]
    # trainset.data = [trainset.data[id] for id in idx]
    # train_sampler,val_sampler = TrainVal_split(trainset, 0.1, shuffle_dataset=True)

    dataset_size = len(trainset.targets)
    args.num_iter = int(0.1*dataset_size) // args.batch_size

    if args.partition == 'class':
        idx = [tgt in class_idx for tgt in trainset.targets]
        idx = np.arange(len(trainset.targets))[idx]
        print(idx[:10])
        trainset.targets = [trainset.targets[id] for id in idx]
        trainset.data = [trainset.data[id] for id in idx]
    elif args.partition == 'hetero-dir':
        # if rank==0:
        net_dataidx_map = partition_data(trainset, args=args)
        trainset.targets = [trainset.targets[id] for id in net_dataidx_map[rank]]
        trainset.data = [trainset.data[id] for id in net_dataidx_map[rank]]
        print(len(trainset.targets))

    train_sampler, val_sampler = TrainVal_split(trainset, 0.1, shuffle_dataset=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2,shuffle=True)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size//size, num_workers=2,sampler=val_sampler)

    testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    net = ResNet_hyper(depth=depth, gate_flag=True, norm_layer=nn.GroupNorm)
    width, structure = net.count_structure()

    # hyper_net = HyperStructure(structure=structure, T=0.4, base=args.base)
    if args.hn_arch == 'simple':
        hyper_net = Simplified_Gate(structure=structure, T=0.4, base=args.base)
    else:
        hyper_net = HyperStructure(structure=structure, T=0.4, base=args.base)


    # if rank==0:
    #     print(hyper_net)
    stat_dict = torch.load('./checkpoint/%s-base.pth.tar'%(model_name+str(depth)))
    net.load_state_dict(stat_dict['net'])
    net.foreze_weights()


    size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnet(net)
    resource_reg = Flops_constraint_resnet(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                        w=args.reg_w, HN=True,structure=structure,)

    Epoch = args.epoch

    hyper_net.cuda()
    net.cuda()
    # filter(lambda p: p.requires_grad, hyper_net.parameters())
    if args.opt == 'AdamW':
        if args.hn_arch == 'simple':
            hyper_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, hyper_net.parameters()), lr=1e-2,
                                          weight_decay=1e-3)
        else:
            hyper_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, hyper_net.parameters()), lr=1e-3,
                                          weight_decay=1e-2)
    elif args.opt == 'Momentum':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, hyper_net.parameters()), lr=0.1, momentum=0.9)
    # print(optimizer)
    scheduler = MultiStepLR(optimizer, milestones=[int(Epoch*0.8)], gamma=0.1)
    best_acc = 0
    if rank == 0:
        valid(0, net, testloader, best_acc, hyper_net=None, model_string=None, stage='valid_model',)

    for epoch in range(Epoch):
        train_hyper(epoch, net, validloader, optimizer, hyper_net=hyper_net, resource_constraint=resource_reg, args=args)
        # train_epm(validloader, net, optimizer, optimizer_p, epoch, args, resource_constraint=resource_reg, hyper_net=hyper_net,
        #         pp_net=pp_net, epm=ep_mem, ep_bn=64, orth_grad=args.orth_grad,use_sampler=args.sampling, loss_type=args.pn_loss)
        scheduler.step()
        if rank == 0:
            best_acc = valid(epoch, net, testloader, best_acc, hyper_net=hyper_net, model_string='%s-pruned'%(model_name+str(depth)), stage='valid_model',)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--opt', default='Momentum', choices=['SGD', 'Momentum', 'Adam', 'AdamW'])
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--stage', default='train-gate', type=str)
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--depth', default=56, type=int)

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--reg_w', default=2, type=float)
parser.add_argument('--base', default=3.0, type=float)
parser.add_argument('--hn_arch', default='hn', type=str)

parser.add_argument('--local_steps', default=5, type=int)
parser.add_argument('--world_size', default=2, type=int)
parser.add_argument('--batch_size', default=256, type=int)

parser.add_argument('--partition', default='hetero-dir',choices=['hetero-dir', 'class', 'homo'])
parser.add_argument('--gpu_visible', default='1', type=str)
# parser.add_argument('--nf', default=1.0, type=float)
# parser.add_argument('--epm_flag', default=False, type=bool)
# parser.add_argument('--loss', default='log', type=str)
# parser.add_argument('--pn_type', default='pn', type=str)
# parser.add_argument('--sampling', default=True, type=str2bool)
# parser.add_argument('--orth_grad',  default=True, type=str2bool)
# parser.add_argument('--pn_loss', default='mae', type=str)
args = parser.parse_args()


size = args.world_size
processes = []

for rank in range(size):
    p = Process(target=init_processes, args=(rank, size, args, run))
    p.start()
    processes.append(p)

for p in processes:
    p.join()