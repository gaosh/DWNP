from asyncio.log import logger
from tqdm import tqdm
import torch
import os
from utils import display_structure, loss_fn_kd, loss_label_smoothing, display_factor, display_structure_hyper, LabelSmoothingLoss
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.sampler import ImbalancedAccuracySampler
from models.hypernet import merge_output
from optimizer import AdamW, Adam
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.nn as nn
import torch
import time

def average_states(args, model, optimizer, weight=None):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    rank = dist.get_rank()
    if weight is None:
        weight = 1 / size
    for param in model.parameters():
        param.requires_grad_(False)
        param *= weight
        dist.all_reduce(param, op=dist.ReduceOp.SUM)
        param.requires_grad_(True)
    # if rank == 0:
    #     print(type(optimizer) == torch.optim.SGD)
    #####Average optimizer parameter
    if type(optimizer) == torch.optim.SGD:
    # if args.opt == 'momentum':
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                state['momentum_buffer'].detach_()
                dist.all_reduce(state['momentum_buffer'])
                state['momentum_buffer'] /= size
                state['momentum_buffer'].requires_grad_()
    elif type(optimizer) == AdamW or type(optimizer) == Adam:
    # elif args.opt == 'Adam' or args.opt == 'AdamW':
        #Todo use better averaging strategy  
        for group in optimizer.param_groups:
            for p in group['params']:
                # print(p.size())
                # print(p.requires_grad)
                if not p.requires_grad:
                    continue



                state = optimizer.state[p]
                # print(state)
                # for key, value in state.items():
                #     print(key)
                state['exp_avg'].detach_()
                dist.all_reduce(state['exp_avg'])
                state['exp_avg'] /= size
                state['exp_avg'].requires_grad_()

                if group['fed']:
                    state['exp_avg_sq'].detach_()
                    dist.all_reduce(state['exp_avg_sq'])
                    state['exp_avg_sq'] /= size
                    state['global_exp_avg_sq'] = state['exp_avg_sq'].clone()
                    state['exp_avg_sq'].requires_grad_()
                else:
                    state['exp_avg_sq'].detach_()
                    dist.all_reduce(state['exp_avg_sq'])
                    state['exp_avg_sq'] /= size
                    state['exp_avg_sq'].requires_grad_()



def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def train_hyper(epoch, net, trainloader, optimizer,hyper_net=None, resource_constraint=None, args=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    net.eval()

    # if rank ==0:
    #     tqdm_loader = tqdm(trainloader)
    # else:
    #     tqdm_loader = trainloader

    if rank ==0:
        tqdm_loader = tqdm(trainloader)
        tqdm_range = tqdm(range(args.gate_iter))
    else:
        tqdm_loader = trainloader
        tqdm_range = range(args.gate_iter)

    hyper_net.train()
    #teacher_net.foreze_weights()
    txtdir = './txt/'

    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    resource_loss = 0
    correct = 0
    total = 0
    # if gw_optim is not None:
    #     backup_res_constraint = copy.deepcopy(resource_constraint)
    for batch_idx in tqdm_range:
        inputs, targets = next(iter(trainloader))
        inputs, targets = inputs.cuda(), targets.cuda()
        # optimizer.zero_grad()

        vector = hyper_net()

        net.set_vritual_gate(vector)

        outputs = net(inputs)

        # loss = criterion(outputs, targets)
        # if stage == 'train_gate':
        res_loss = 2 * resource_constraint(hyper_net.resource_output())
        loss = criterion(outputs, targets)

        loss = loss + res_loss
        # else:
        #     loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        # optimizer.step(update_adapt_lr=False)
        optimizer.step()

        if (batch_idx+1) % args.local_steps == 0:
            average_states(args, hyper_net, optimizer)
            # optimizer.step(update_momentum=False, update_adapt_lr=True, update_params=False)

        train_loss += loss.detach().clone()
        resource_loss += res_loss.detach().clone()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().detach().clone()


    train_loss /=  len(trainloader)
    resource_loss /= len(trainloader)
    accuracy = 100. * correct/ total

    dist.all_reduce(train_loss)
    dist.all_reduce(resource_loss)
    dist.all_reduce(accuracy)
    dist.all_reduce(correct)

    train_loss = train_loss.item()/world_size
    resource_loss = resource_loss.item()/world_size
    accuracy = accuracy.item()/world_size

    if rank == 0:
        print('Epoch: %d Loss: %.3f Res-Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (epoch, train_loss, resource_loss, accuracy, correct, total*world_size))

    #with torch.no_grad():
    #    print_string = display_structure(hyper_net.transfrom_output(vector))

        with torch.no_grad():
            vector = hyper_net()
            display_structure_hyper(hyper_net.transfrom_output(vector))

        with torch.no_grad():
            resource_constraint.print_current_FLOPs(hyper_net.resource_output())

    return resource_loss / len(trainloader)

def one_step_hypernet(inputs, targets, net, hyper_net, args, dynamic_emb=None):
    net.eval()
    hyper_net.train()

    # inputs, targets = inputs.cuda(), targets.cuda()
    # optimizer.zero_grad()

    vector = hyper_net()
    if dynamic_emb is not None:
        dynamic_emb.train()
        if args.partition == 'class':
            task_id = targets
            dynamic_vector = dynamic_emb(task_id)
        else:
            task_id = dist.get_rank()
            task_id = torch.Tensor([task_id]).long().cuda()
            dynamic_vector = dynamic_emb(task_id)
        dynamic_vector = merge_output(vector, dynamic_vector)
        vector = vector*dynamic_vector
    net.set_vritual_gate(vector)

    outputs = net(inputs)

    res_loss = 2 * args.resource_constraint(hyper_net.resource_output())
    loss = nn.CrossEntropyLoss()(outputs, targets) + res_loss
    if dynamic_emb is not None:
        discrete_vector = hyper_net.resource_output()
        discrete_dynamic_vector = dynamic_emb.resource_output(task_id)
        discrete_dynamic_vector = merge_output(discrete_vector, discrete_dynamic_vector)
        discrete_vector = discrete_vector.detach() * discrete_dynamic_vector
        res_loss_dynamic = 2 * args.resource_constraint_dynamic(discrete_vector)
        loss += res_loss_dynamic
        loss.backward()

        return loss, res_loss, res_loss_dynamic, outputs
    else:
        loss.backward()
        return loss, res_loss, outputs

def one_step_net(inputs, targets, net, hyper_net, args, dynamic_emb=None):
    net.train()
    with torch.no_grad():
        hyper_net.eval()
        vector = hyper_net()
        if dynamic_emb is not None:
            if args.partition == 'class':
                task_id = targets
                dynamic_emb.eval()
                dynamic_vector = dynamic_emb(task_id)
            else:
                task_id = dist.get_rank()
                dynamic_emb.eval()
                task_id = torch.Tensor([task_id]).long().cuda()
                dynamic_vector = dynamic_emb(task_id)
            vector = vector * dynamic_vector

        net.set_vritual_gate(vector)

    # inputs, targets = inputs.cuda(), targets.cuda()
    outputs = net(inputs)

    if args.smooth_flag:
        loss_smooth = LabelSmoothingLoss(classes=10, smoothing=0.1)(outputs, targets)
        loss_c = nn.CrossEntropyLoss()(outputs, targets)
        loss = args.alpha * loss_smooth + (1 - args.alpha) * loss_c
    else:
        loss = nn.CrossEntropyLoss()(outputs, targets)

    loss.backward()

    return loss, outputs

def iterative_dynamic_train(epoch, models, optimizers, dataloaders, args=None, logger=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()


    train_loss = 0
    correct = 0
    total = 0

    hyper_loss=0
    h_correct = 0
    h_total = 0
    resource_loss = 0
    resource_loss_dynamic = 0
    h_num = 0

    net = models['net']
    hyper_net = models['hyper_net']
    dynamic_emb = models['dynamic_emb']
    hyper_optimizer = optimizers['hyper_optimizer']
    emb_optimizer = optimizers['emb_optimizer']
    optimizer = optimizers['net_optimizer']

    net.train()

    trainloader = dataloaders['train']
    validloader = dataloaders['valid']

    if rank == 0:
        total_time = 0
        communication_time = 0

    if rank ==0:
        tqdm_loader = tqdm(trainloader)
        tqdm_range = tqdm(range(args.num_iter))
    else:
        tqdm_loader = trainloader
        tqdm_range = range(args.num_iter)

    numBatches = len(trainloader)

    # for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
    for batch_idx in tqdm_range:
        inputs, targets = next(iter(trainloader))
        inputs, targets = inputs.cuda(), targets.cuda()
        if rank == 0: time0 = time.time()

        # if scheduler is not None:
        #     scheduler.step()
        # inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        loss, outputs = one_step_net(inputs, targets, net, hyper_net, args, dynamic_emb=dynamic_emb)
        optimizer.step()

        if rank == 0: time2 = time.time()
        if (batch_idx+1) % args.local_steps == 0:
            average_states(args, net, optimizer)
        if rank == 0: time3 = time.time()

        train_loss += loss.detach().clone()
        _, predicted = outputs.max(1)
        total += torch.tensor(targets.size(0)).cuda()
        correct += predicted.eq(targets).sum().detach().clone()

        if epoch >= args.start_epoch  and (batch_idx + 1) % args.hyper_interval == 0:
            for _ in range(args.hyper_steps):
                val_inputs, val_targets = next(iter(validloader))
                val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()

                hyper_optimizer.zero_grad()
                emb_optimizer.zero_grad()
                h_loss, res_loss, res_loss_dynamic ,hyper_outputs = one_step_hypernet(val_inputs, val_targets, net, hyper_net, args, dynamic_emb=dynamic_emb)

                hyper_optimizer.step()
                emb_optimizer.step()
                hyper_loss += h_loss.detach().clone()
                resource_loss += res_loss.detach().clone()
                resource_loss_dynamic += res_loss_dynamic.detach().clone()
                h_num += 1

                _, h_predicted = hyper_outputs.max(1)
                h_total += torch.tensor(val_targets.size(0)).cuda()
                h_correct += h_predicted.eq(val_targets).sum().detach().clone()



        if rank == 0: time1 = time.time()

        if epoch >= args.start_epoch and (batch_idx + 1) % args.local_steps == 0:
            average_states(args, hyper_net, hyper_optimizer)
            average_states(args, dynamic_emb, emb_optimizer)

        if rank == 0:
            total_time += time.time() - time0
            communication_time += time.time() - time1 + time3 - time2



    average_states(args, net, optimizer)

    if epoch >= args.start_epoch:
        average_states(args, hyper_net, hyper_optimizer)
        average_states(args, dynamic_emb, emb_optimizer)

    train_loss /= len(trainloader)
    accuracy = 100. * correct / total

    if h_num > 0:
        hyper_loss /= h_num
        resource_loss /= h_num
        resource_loss_dynamic /= h_num
        h_accuracy = 100. * h_correct / h_total
    else:
        hyper_loss = torch.tensor(-1).cuda()
        resource_loss = torch.tensor(-1).cuda()
        resource_loss_dynamic = torch.tensor(-1).cuda()
        h_accuracy = torch.tensor(-1).cuda()
        h_correct = torch.tensor(-1).cuda()
        h_total = torch.tensor(-1).cuda()

    dist.all_reduce(train_loss)
    dist.all_reduce(accuracy)
    dist.all_reduce(correct)
    dist.all_reduce(total)
    dist.all_reduce(h_correct)
    dist.all_reduce(h_total)
    dist.all_reduce(hyper_loss)
    dist.all_reduce(resource_loss)
    dist.all_reduce(resource_loss_dynamic)
    dist.all_reduce(h_accuracy)

    train_loss = train_loss.item() / world_size
    accuracy = accuracy.item() / world_size
    hyper_loss = hyper_loss.item() / world_size
    resource_loss = resource_loss.item() / world_size
    resource_loss_dynamic = resource_loss_dynamic.item() / world_size
    h_accuracy = h_accuracy.item() / world_size


    if rank == 0:
        with torch.no_grad():
            vector = hyper_net()
            display_structure_hyper(hyper_net.transfrom_output(vector))

        with torch.no_grad():
            args.resource_constraint.print_current_FLOPs(hyper_net.resource_output())

        # print('Epoch: %d Loss: %.3f Res-Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #       % (epoch, train_loss, resource_loss, accuracy, correct, total * world_size))

        logger.update('tr_samples', total.item())
        logger.update('tr_correct_samples', correct.item())
        logger.update('tr_hyper_samples', h_total.item())
        logger.update('tr_hyper_correct_samples', h_correct.item())
        logger.update('tr_loss', train_loss)
        logger.update('tr_acc', accuracy)
        logger.update('tr_hyper_loss', hyper_loss)
        logger.update('tr_hyper_res_loss', resource_loss)
        logger.update('tr_hyper_res_dynamic_loss', resource_loss_dynamic)
        logger.update('tr_hyper_acc', h_accuracy)
        logger.update('comm_time', communication_time)
        logger.update('tr_time', total_time)
        logger.update('comm_percent', round(communication_time/total_time, 3))



def iterative_train(epoch, net, trainloader, optimizer, hyper_net, validloader, hyper_optimizer, args=None, logger=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    hyper_loss=0
    h_correct = 0
    h_total = 0
    resource_loss = 0
    h_num = 0

    if rank == 0:
        total_time = 0
        communication_time = 0

    if rank ==0:
        tqdm_loader = tqdm(trainloader)
        tqdm_range = tqdm(range(args.num_iter))
    else:
        tqdm_loader = trainloader
        tqdm_range = range(args.num_iter)

    numBatches = len(trainloader)

    # for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
    for batch_idx in tqdm_range:
        inputs, targets = next(iter(trainloader))
        inputs, targets = inputs.cuda(), targets.cuda()
        if rank == 0: time0 = time.time()

        # if scheduler is not None:
        #     scheduler.step()
        # inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        loss, outputs = one_step_net(inputs, targets, net, hyper_net, args)
        optimizer.step()

        if rank == 0: time2 = time.time()
        if (batch_idx+1) % args.local_steps == 0:
            average_states(args, net, optimizer)
        if rank == 0: time3 = time.time()

        train_loss += loss.detach().clone()
        _, predicted = outputs.max(1)
        total += torch.tensor(targets.size(0)).cuda()
        correct += predicted.eq(targets).sum().detach().clone()

        if epoch >= args.start_epoch  and (batch_idx + 1) % args.hyper_interval == 0:
            for _ in range(args.hyper_steps):
                val_inputs, val_targets = next(iter(validloader))
                val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()

                hyper_optimizer.zero_grad()
                
                h_loss, res_loss,hyper_outputs = one_step_hypernet(val_inputs, val_targets, net, hyper_net, args)

                hyper_optimizer.step()

                hyper_loss += h_loss.detach().clone()
                resource_loss += res_loss.detach().clone()
                h_num += 1

                _, h_predicted = hyper_outputs.max(1)
                h_total += torch.tensor(val_targets.size(0)).cuda()
                h_correct += h_predicted.eq(val_targets).sum().detach().clone()

            if (batch_idx + 1) % args.local_steps == 0:
                average_states(args, hyper_net, hyper_optimizer)



        if rank == 0: time1 = time.time()


        if rank == 0: 
            total_time += time.time() - time0
            communication_time += time.time() - time1 + time3 - time2



    average_states(args, net, optimizer)
    if epoch >= args.start_epoch:
        average_states(args, hyper_net, hyper_optimizer)

    train_loss /= len(trainloader)
    accuracy = 100. * correct / total

    if h_num > 0:
        hyper_loss /= h_num
        resource_loss /= h_num
        h_accuracy = 100. * h_correct / h_total
    else:
        hyper_loss = torch.tensor(-1).cuda()
        resource_loss = torch.tensor(-1).cuda()
        h_accuracy = torch.tensor(-1).cuda()
        h_correct = torch.tensor(-1).cuda()
        h_total = torch.tensor(-1).cuda()

    dist.all_reduce(train_loss)
    dist.all_reduce(accuracy)
    dist.all_reduce(correct)
    dist.all_reduce(total)
    dist.all_reduce(h_correct)
    dist.all_reduce(h_total)
    dist.all_reduce(hyper_loss)
    dist.all_reduce(resource_loss)
    dist.all_reduce(h_accuracy)

    train_loss = train_loss.item() / world_size
    accuracy = accuracy.item() / world_size
    hyper_loss = hyper_loss.item() / world_size
    resource_loss = resource_loss.item() / world_size
    h_accuracy = h_accuracy.item() / world_size


    if rank == 0:
        with torch.no_grad():
            vector = hyper_net()
            display_structure_hyper(hyper_net.transfrom_output(vector))

        with torch.no_grad():
            args.resource_constraint.print_current_FLOPs(hyper_net.resource_output())

        # print('Epoch: %d Loss: %.3f Res-Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #       % (epoch, train_loss, resource_loss, accuracy, correct, total * world_size))

        logger.update('tr_samples', total.item())
        logger.update('tr_correct_samples', correct.item())
        logger.update('tr_hyper_samples', h_total.item())
        logger.update('tr_hyper_correct_samples', h_correct.item())
        logger.update('tr_loss', train_loss)
        logger.update('tr_acc', accuracy)
        logger.update('tr_hyper_loss', hyper_loss)
        logger.update('tr_hyper_res_loss', resource_loss)
        logger.update('tr_hyper_acc', h_accuracy)
        logger.update('comm_time', communication_time)
        logger.update('tr_time', total_time)
        logger.update('comm_percent', round(communication_time/total_time, 3))


def retrain(epoch, net,trainloader, optimizer, smooth=True, scheduler=None, alpha=0.5, hyper_net=None, args=None, logger=None):
    #net.activate_weights()
    #net.set_training_flag(False)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank ==0: 
        tqdm_loader = tqdm(trainloader)
        tqdm_range = tqdm(range(args.num_iter))
    else:
        tqdm_loader = trainloader
        tqdm_range = range(args.num_iter)


    net.train()
    train_loss = 0
    correct = 0
    total = 0
    sum_res_dynamic = 0
    alpha = alpha
    criterion = nn.CrossEntropyLoss()

    if hyper_net is not None and (epoch >= args.start_epoch):
        with torch.no_grad():
            hyper_net.eval()
            vector = hyper_net()
            net.set_vritual_gate(vector)

    # for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
    for batch_idx in tqdm_range:
        if scheduler is not None:
            scheduler.step()
        inputs, targets = next(iter(trainloader))
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        if args.method == 'dynamic':
            task_id = rank
            client_vector = args.dynamic_cfg[task_id]
            net.set_vritual_gate(client_vector.cuda())

        elif args.method == 'dynamic_train':
            args.dynamic_emb.train()
            task_id = rank
            task_id = torch.Tensor([task_id]).long().cuda()
            dynamic_vector =  args.dynamic_emb(task_id)
            dynamic_vector = args.dynamic_emb.prune_output(dynamic_vector)
            net.set_vritual_gate(dynamic_vector)

            discrete_dynamic_vector = args.dynamic_emb.resource_output(task_id)
            discrete_dynamic_vector = args.dynamic_emb.prune_output(discrete_dynamic_vector)
            res_loss_dynamic = 2 * args.resource_constraint_dynamic(discrete_dynamic_vector)

            args.dynamic_optim.zero_grad()
        outputs = net(inputs)
        if smooth:
            loss_smooth = LabelSmoothingLoss(classes=10,smoothing=0.1)(outputs, targets)
            loss_c = criterion(outputs, targets)
            loss = alpha*loss_smooth + (1-alpha)*loss_c
        else:
            loss = criterion(outputs, targets)

        if args.method == 'dynamic_train':
            loss += res_loss_dynamic
            loss.backward()
            optimizer.step()
            args.dynamic_optim.step()
        else:
            loss.backward()
            optimizer.step()

        if (batch_idx+1) % args.local_steps == 0:
            average_states(args, net, optimizer)
            if args.method == 'dynamic_train':
                average_states(args, args.dynamic_emb, args.dynamic_optim)
        train_loss += loss.detach().clone()
        _, predicted = outputs.max(1)
        total += torch.tensor(targets.size(0)).cuda()
        correct += predicted.eq(targets).sum().detach().clone()
        if args.method == 'dynamic_train':
            sum_res_dynamic += res_loss_dynamic.detach().clone()

    average_states(args, net, optimizer)
    if args.method == 'dynamic_train':
        average_states(args, args.dynamic_emb, args.dynamic_optim)

    train_loss /=  len(trainloader)
    sum_res_dynamic /= len(trainloader)
    accuracy = 100. * correct/ total

    dist.all_reduce(train_loss)
    dist.all_reduce(accuracy)
    dist.all_reduce(correct)
    dist.all_reduce(total)
    if args.method == 'dynamic_train':
        dist.all_reduce(sum_res_dynamic)
        sum_res_dynamic = sum_res_dynamic.item()/world_size
    train_loss = train_loss.item()/world_size
    accuracy = accuracy.item()/world_size

    if rank == 0 and logger is not None:
        logger.update('tr_samples', total.item())
        logger.update('tr_correct_samples', correct.item())
        logger.update('tr_loss', train_loss)
        logger.update('tr_acc', accuracy)
        if args.method == 'dynamic_train':
            logger.update('tr_hyper_res_dynamic_loss', sum_res_dynamic)

def distributed_vaild(epoch, net, testloader, best_acc, hyper_net=None, dynamic_emb=None, dynamic_cfg=None, model_string=None, stage='valid_model', logger=None, args=None):
#     if stage == 'valid_model':
#         # tqdm_loader = tqdm(testloader)
#         tqdm_loader = testloader
#     elif stage == 'valid_gate':
#         #net.foreze_weights()
#         if hyper_net is None:
#             net.set_training_flag(True)

    tqdm_loader = testloader
    criterion = torch.nn.CrossEntropyLoss()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    net.eval()
    if rank == 0:
        tqdm_loader = tqdm(tqdm_loader)

    if hyper_net is not None:
        hyper_net.eval()
        vector = hyper_net()
        # print(vector)
        net.set_vritual_gate(vector)
    test_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # with torch.nn.parallel.distributed.join():
            if dynamic_emb is not None:
                if args.partition == 'class':
                    task_id = targets
                else:
                    # task_id = torch.randint(0, args.world_size, targets.size()).long().cuda()
                    task_id = dist.get_rank()
                    task_id = torch.Tensor([task_id]).long().cuda()
                dynamic_emb.eval()
                dynamic_vector = dynamic_emb(task_id)

                if args.method == 'dynamic_train':
                    vector = dynamic_emb.prune_output(dynamic_vector)
                else:
                    vector = vector * dynamic_vector
                net.set_vritual_gate(vector)

            if dynamic_cfg is not None:
                task_id = rank
                client_vector = dynamic_cfg[task_id]
                net.set_vritual_gate(client_vector.cuda())

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss

            _, predicted = outputs.max(1)

            total += torch.tensor(targets.size(0)).cuda()
            correct += predicted.eq(targets).sum()


    dist.barrier()
    test_loss /= len(testloader)
    dist.all_reduce(test_loss)
    dist.all_reduce(correct)
    dist.all_reduce(total)
    test_loss = test_loss.item()/world_size
    if rank == 0:
        correct = correct.item()
        total = total.item()
        acc = 100.*correct/total
        is_best=False
        if hyper_net is not None:
            if epoch>100:
                if acc > best_acc:
                    best_acc = acc
                    is_best=True
            else:
                best_acc = 0
        else:
            if acc>best_acc:
                best_acc = acc
                is_best = True

        if model_string is not None:

            if is_best:
                print('Saving..')
                if hyper_net is not None and dynamic_emb is None:

                    state = {
                        'net': net.state_dict(),
                        'hyper_net': hyper_net.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                        'arch_vector': vector
                        # 'gpinfo':gplist,
                    }
                elif hyper_net is not None and dynamic_emb is not None:
                    state = {
                        'net': net.state_dict(),
                        'hyper_net': hyper_net.state_dict(),
                        'dynamic_emb': dynamic_emb.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                        'arch_vector': vector
                        # 'gpinfo':gplist,
                    }
                else:
                    if dynamic_cfg is not None:
                        state = {
                            'net': net.state_dict(),
                            'acc': acc,
                            'epoch': epoch,
                            'dynamic_cfg':dynamic_cfg,
                            # 'gpinfo':gplist,
                        }
                    elif dynamic_emb is not None:
                        state = {
                            'net': net.state_dict(),
                            'acc': acc,
                            'epoch': epoch,
                            'dynamic_emb': dynamic_emb.state_dict(),
                            # 'gpinfo':gplist,
                        }
                    else:
                        state = {
                            'net': net.state_dict(),
                            'acc': acc,
                            'epoch': epoch,
                            # 'gpinfo':gplist,
                        }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')

                torch.save(state, './checkpoint/%s.pth.tar' % (model_string))

        if logger is not None:
            logger.update('test_samples', total)
            logger.update('test_correct_samples', correct)
            logger.update('test_loss', test_loss)
            logger.update('test_acc', 100. * correct / total)
            logger.update('best_test_acc', best_acc)

        print('Epoch %d Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc: %.3f%%'
              % (epoch, test_loss, 100. * correct / total, correct, total, best_acc))
        return best_acc

def valid(epoch, net, testloader, best_acc, hyper_net=None, dynamic_emb=None, model_string=None, stage='valid_model', logger=None, args=None):
    txtdir = './txt/'
    if stage == 'valid_model':
        # tqdm_loader = tqdm(testloader)
        tqdm_loader = testloader
    elif stage == 'valid_gate':
        #net.foreze_weights()
        if hyper_net is None:
            net.set_training_flag(True)
        tqdm_loader = testloader
    criterion = torch.nn.CrossEntropyLoss()

    net.eval()
    if hyper_net is not None and (epoch >= args.start_epoch):
        with torch.no_grad():
            hyper_net.eval()
            vector = hyper_net()
            # print(vector)
            net.set_vritual_gate(vector)

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            if dynamic_emb is not None:
                if args.partition == 'class':
                    task_id = targets
                else:
                    task_id = torch.randint(0, args.world_size, targets.size()).long().cuda()
                dynamic_emb.eval()
                dynamic_vector = dynamic_emb(task_id)

                vector = vector * dynamic_vector
                net.set_vritual_gate(vector)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    acc = 100.*correct/total
    is_best=False
    if hyper_net is not None:
        if epoch>100:
            if acc > best_acc:
                best_acc = acc
                is_best=True
        else:
            best_acc = 0
    else:
        if acc>best_acc:
            best_acc = acc
            is_best = True
    if model_string is not None:

        if is_best:
            print('Saving..')
            if hyper_net is not None:

                state = {
                    'net': net.state_dict(),
                    'hyper_net': hyper_net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'arch_vector':vector
                    #'gpinfo':gplist,
                }
            elif dynamic_emb is not None:
                state = {
                    'net': net.state_dict(),
                    'hyper_net': hyper_net.state_dict(),
                    'dynamic_emb': dynamic_emb.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'arch_vector': vector
                    # 'gpinfo':gplist,
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    # 'gpinfo':gplist,
                }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/%s.pth.tar'%(model_string))

    if logger is not None:
        logger.update('test_samples', total)
        logger.update('test_correct_samples', correct)
        logger.update('test_loss', test_loss/len(testloader))
        logger.update('test_acc', 100.*correct/total)
        logger.update('best_test_acc', best_acc)

    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc: %.3f%%'
          % (test_loss / len(testloader), 100. * correct / total, correct, total, best_acc))

    return best_acc


