from train import *
from utils import *
from repeat_dataloader import MultiEpochsDataLoader
# from models.vgg_gate import *
from models.resnet_hyper import ResNet
#from models.mobilenetv2 import MobileNetV2
from models.mobilenetv2_hyper import MobileNetV2
from models.gate_function import *
from torch.optim.lr_scheduler import MultiStepLR
from models.hypernet import DynamicEmbedding
from data_util import partition_data
import torch.nn as nn

import argparse
import torchvision
import torchvision.transforms as transforms

from warm_up.Warmup_Sch import GradualWarmupScheduler

from torch.multiprocessing import Process
import torch.distributed as dist
from optimizer import AdamW
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

# parser.add_argument('--model_name', default='resnet', type=str)
parser.add_argument('--depth',  default=56, type=int)
parser.add_argument('--gpu_visible', default='1', type=str)

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--m', default=0.9, type=float)
parser.add_argument('--sch', default='first-more',type=str)
parser.add_argument('--smooth_flag', default=False, type=str2bool)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--warmup', default=False, type=str2bool)
parser.add_argument('--train_base', default=True, type=str2bool)
parser.add_argument('--local_steps', default=5, type=int)
parser.add_argument('--world_size', default=2, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--opt', default='Momentum', choices=['SGD', 'Momentum', 'Adam', 'AdamW'])
parser.add_argument('--partition', default='hetero-dir',choices=['hetero-dir', 'class', 'homo'])
parser.add_argument('--model_name', default='resnet56',type=str)

parser.add_argument('--method', default='dynamic_train',choices=['static', 'dynamic', 'dynamic_train'])
parser.add_argument('--split_test', default=True, type=str2bool)
parser.add_argument('--d_p', default=0.5, type=float)
args = parser.parse_args()


print(args.train_base)

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


    prefix = '-fine-tune'
    logger = Logger('results', prefix)


    class_idx = set(np.arange(cls_per_client*rank,cls_per_client*(rank+1)))

    trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=False, transform=transform_train)
    # if rank == 0:
    testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=False,
                                           transform=transform_test)
    dataset_size = len(trainset.targets)
    args.num_iter = dataset_size//args.batch_size

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
            net_dataidx_map, _ = partition_data(trainset, args=args, test_dataset=testset)
            # net_dataidx_map = partition_data(trainset, args=args)
            trainset.targets = [trainset.targets[id] for id in net_dataidx_map[rank]]
            trainset.data = [trainset.data[id] for id in net_dataidx_map[rank]]
            print(len(trainset.targets))
    # train_sampler,val_sampler = TrainVal_split(trainset, 0.1, shuffle_dataset=True)
    trainloader = MultiEpochsDataLoader(trainset, batch_size=args.batch_size // size, num_workers=1,shuffle=True, pin_memory=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=200// size, shuffle=False, num_workers=1)

    if args.model_name == 'resnet56' or args.model_name == 'mobnetv2':
        if args.train_base:
            net = ResNet(depth=args.depth, gate_flag=True, norm_layer=nn.GroupNorm)
            model_name = '%s-base' % (args.model_name)
        else:
            # stat_dict = torch.load('./checkpoint/%s_new.pth.tar' % (args.model_name))
            if args.method == 'static':
                stat_dict = torch.load('./checkpoint/%s_%s_new.pth.tar'%(args.model_name, args.method))
            elif args.method == 'dynamic' or  args.method == 'dynamic_train':
                stat_dict = torch.load('./checkpoint/%s_new.pth.tar' % (args.model_name))
            else:
                stat_dict = torch.load('./checkpoint/%s_%s_new.pth.tar' % (args.model_name, args.method))
            if args.model_name == 'resnet56':
                net = ResNet(depth=args.depth, cfg=stat_dict['cfg'], norm_layer=nn.GroupNorm)
                ori_net = ResNet(depth=args.depth, gate_flag=True, norm_layer=nn.GroupNorm)
            elif args.model_name == 'mobnetv2':
                net = MobileNetV2(cfg=stat_dict['cfg'], norm_layer=nn.GroupNorm)
                ori_net = MobileNetV2(norm_layer=nn.GroupNorm)

            net.load_state_dict(stat_dict['state_dict'])
            if args.method == 'dynamic':
                dynamic_cfg = stat_dict['dynamic_cfg']
                args.dynamic_cfg = dynamic_cfg
                dynamic_emb = None
            elif args.method == 'dynamic_train':
                dynamic_cfg = None

                width, structure = ori_net.count_structure()
                net.count_structure()
                if args.model_name == 'mobnetv2':
                    size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_mobnet(ori_net)
                    resource_reg = Flops_constraint_mobnet(args.d_p, size_kernel, size_out, size_group, size_inchannel,
                                                           size_outchannel,
                                                           w=2.0, HN=True, structure=structure, )
                else:
                    size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnet(ori_net)
                    resource_reg = Flops_constraint_resnet(args.d_p, size_kernel, size_out, size_group, size_inchannel,
                                                       size_outchannel,
                                                       w=2.0, HN=True, structure=structure, )
                args.resource_constraint_dynamic = resource_reg
                dynamic_emb = DynamicEmbedding(structure=structure, T=0.4, base=3.0, num_clients=args.world_size)
                dynamic_emb.cuda()
                dynamic_emb.load_state_dict(stat_dict['dynamic_emb'])
                dynamic_emb.remain_index = stat_dict['pruned_index']
                args.dynamic_emb = dynamic_emb
                emb_params = list(filter(lambda p: p.requires_grad, dynamic_emb.parameters()))
                emb_optimizer = AdamW(emb_params, lr=1e-3, weight_decay=1e-2)
                args.dynamic_optim = emb_optimizer

            model_name = '%s-ft' % (args.model_name)
    # elif args.model_name == 'mobnetv2':
    #     if args.train_base:
    #         net = MobileNetV2(gate_flag=True)
    #         model_name = '%s-base' % (args.model_name)
    #     else:
    #         stat_dict = torch.load('./checkpoint/%s_new.pth.tar' % (args.model_name))
    #         net = MobileNetV2(custom_cfg=stat_dict['cfg'])
    #         net.load_state_dict(stat_dict['state_dict'])
    #         model_name = '%s-ft' % (args.model_name)

    if args.opt == 'Momentum':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                # momentum=0.9)
                                momentum=args.m, weight_decay=args.wd)
    elif args.opt == 'AdamW':
        optimizer = AdamW(net.parameters(), lr=args.lr,
                                weight_decay=1e-2, fed=False)

    Epoch = args.epoch
    print(Epoch)
    print(type(optimizer))
    net.cuda()

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

    if args.split_test:
        best_acc = 0
        if rank == 0:
            best_acc = distributed_vaild(0, net, testloader, best_acc, hyper_net=None,
                                         dynamic_cfg=dynamic_cfg,
                                         dynamic_emb=dynamic_emb,
                                         model_string='%s-pruned-dynamic' % (model_name),
                                         stage='valid_model',
                                         logger=logger, args=args)
        else:
            distributed_vaild(0, net, testloader, best_acc, hyper_net=None,
                              dynamic_cfg=dynamic_cfg,
                              dynamic_emb=dynamic_emb,
                              model_string='%s-pruned-dynamic' % (model_name),
                              stage='valid_model',
                              logger=logger, args=args)
    else:
        if rank == 0:
            best_acc = 0
            best_acc = valid(0, net, testloader, best_acc, hyper_net=None, model_string=model_name, logger=logger)
            print(scheduler)


    for epoch in range(0, Epoch):
        #scheduler.step()

        scheduler.step()
        if rank == 0:
            print(scheduler.get_lr())
        retrain(epoch, net, trainloader, optimizer, smooth=args.smooth_flag,alpha=args.alpha, args=args, logger=logger)

        if args.split_test:
            if rank == 0:
                best_acc = distributed_vaild(epoch, net, testloader, best_acc, hyper_net=None,
                                             dynamic_cfg=dynamic_cfg,
                                             dynamic_emb=dynamic_emb,
                                             model_string='%s-pruned-dynamic' % (model_name),
                                             stage='valid_model',
                                             logger=logger, args=args)
            else:
                distributed_vaild(epoch, net, testloader, best_acc, hyper_net=None,
                                  dynamic_cfg=dynamic_cfg,
                                  dynamic_emb=dynamic_emb,
                                  model_string='%s-pruned-dynamic' % (model_name),
                                  stage='valid_model',
                                  logger=logger, args=args)
        else:

            if rank == 0:
                best_acc = valid(epoch, net, testloader, best_acc, hyper_net=None, model_string =model_name, logger=logger)

        if rank == 0:
            if args.method == 'dynamic_train':
                logger.print_model_dynamic(epoch)
            else:
                logger.print_model(epoch)
            if epoch % 5 == 0:
                logger.save()


size = args.world_size
processes = []

for rank in range(size):
    p = Process(target=init_processes, args=(rank, size, args, run))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
