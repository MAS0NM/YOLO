import argparse
from audioop import reverse
from crypt import methods

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *
from utils.read_write import write_pruning_rate

####### search ########
from search.search_filter_pruning import random_can_fp
from search.meter import *
import copy
import torch_pruning as tp
import numpy as np
from train_fun import train_func
from train_final import train_finalc
import pandas as pd
import operator

# warmup = True
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

# Hyperparameters
hyp = {
       'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
       'lr0': 0.001,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        # 'optimizer': 'adam',  # ['adam', 'SGD', None] if none, default is SGD
        # 'lr0': 0.0001,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum/Adam beta1
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.5,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)

savelog = False


def select_candidate(global_candidates, interval, reverse=True):

    global_candidates = sorted(global_candidates, key=lambda candidator: candidator[-1], reverse=reverse)
    global_candidates = np.array(global_candidates)
    candidates = global_candidates[::interval,0]
    candidates_acc = global_candidates[::interval,1]

    print('-----------------Select Gen by %s ------------------' %('acc' if reverse else 'loss'))
    for candidator, score in zip(candidates,candidates_acc):
        print('FLOPs = {:.2f}M |'.format(2 * candidator[-1] / 1000000), flush=True, end=' ')
        print('PARAMs = {:.2f}M |'.format(candidator[-2] / 1000000), flush=True, end=' ')
        print('Score = {:.4f} '.format(float(score)))
    print('-------------------------------------------')

    return candidates, candidates_acc



def Adaptive_BN(model, trainLoader):
    losses = utils.AverageMeter()
    model.train()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(trainLoader):
            if batch <= 300:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_func(output, targets)
                losses.update(loss.item(), inputs.size(0))
            else:
                break
    return model, losses.avg

def filter_prune_L1(model, example_inputs, output_transform, pruning_rates):
    # import copy
    # model=copy.deepcopy(model_ref)
    # model.cpu().eval()
    excluded_layers = list(model.model[-1].modules())

    prunable_module_type = (nn.Conv2d)
    prunable_modules = [ m for m in model.modules() if isinstance(m, prunable_module_type) and m not in excluded_layers]
    print('len ', len(prunable_modules))
    print('len ', len(pruning_rates))
    strategy = tp.strategy.L2Strategy()
    # strategy = tp.strategy.L1Strategy()
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs, output_transform=output_transform)
    for layer_to_prune, fp_rate in zip(prunable_modules,pruning_rates):
        # select a layer

        # print(layer_to_prune)
        if isinstance( layer_to_prune, nn.Conv2d ):
            prune_fn = tp.prune_conv

        weight = layer_to_prune.weight.detach().cpu()
        prune_index = strategy(weight, amount=fp_rate)
        plan = DG.get_pruning_plan( layer_to_prune, prune_fn, prune_index)
        plan.exec()
    # with torch.no_grad():
    #     out = model( example_inputs )
    #     if output_transform:
    #         out = output_transform(out)
    # return model


def train(hyp, tb_writer, opt, device):
    print(f'Hyperparameters {hyp}')


    ##################################
    #####   init some hyp params #####
    ##################################
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.local_rank
    # TODO: Init DDP logging. Only the first process is allowed to log.
    # Since I see lots of print here, the logging configuration is skipped here. We may see repeated outputs.

    ##################################
    ####     load data.yaml       ####
    ##################################
    # Configure
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check


    ##################################
    ####       Create model       ####
    ##################################
    # Create model
    # model = Model(opt.cfg, nc=nc).to(device)
    model = Model(opt.cfg, nc=nc)

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    ####################################
    ##   load model from model.yaml   ##
    ####################################
    # Load Model
    with torch_distributed_zero_first(rank):
        google_utils.attempt_download(weights)
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'):  # pytorch format
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            exclude = ['anchor']  # exclude keys
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if k in model.state_dict() and not any(x in k for x in exclude)
                             and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(ckpt['model'], strict=False)
            print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), weights))
        except KeyError as e:
            s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                % (weights, opt.cfg, weights, weights)
            raise KeyError(s) from e

        # # load optimizer
        # if ckpt['optimizer'] is not None:
        #     optimizer.load_state_dict(ckpt['optimizer'])
        #     best_fitness = ckpt['best_fitness']

        # # load results
        # if ckpt.get('training_results') is not None:
        #     with open(results_file, 'a') as file:
        #         file.write(ckpt['training_results'])  # write results.txt

        # # epochs
        # start_epoch = ckpt['epoch'] + 1
        # if epochs < start_epoch:
        #     print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
        #           (weights, ckpt['epoch'], epochs))
        #     epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    
    ###############################
    #####   dateset loader   ######
    ###############################
    # # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                            cache=opt.cache_images, rect=opt.rect, local_rank=rank,
                                            world_size=opt.world_size)

    # np.concatenate 一次完成多个数组的拼接
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    # nb = len(dataloader)  # number of batches, num_training_pics / batch_size
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)


    testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt, hyp=hyp, augment=False,
                                       cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size)[0]


    # Model parameters
    # nc: the number of classes
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Class frequency
    if rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        if savelog:
            plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
            tb_writer.add_histogram('classes', c, 0)

        # Check anchors
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)


    #################################################
    # Torch Pruning (Begin)
    #################################################

    print('start pruning.....')


    # prepare candidates
    prunable_module_type = (nn.Conv2d)
    prunable_modules = [m for m in model.modules() if isinstance(m, prunable_module_type)]
    FP_Rate_Len = len(prunable_modules)
    FP_Rate_Len -= 3
    print('Number of prunable convolution ', FP_Rate_Len)


    layer_params = calc_model_parameters(model)
    layer_flops = calc_model_flops(model, input_size=imgsz, mul_add=False)
    total_params = sum(layer_params)
    total_flops = sum(layer_flops)
    # s_total_flops = '\nModel FLOPs = {:.2f}M'.format(total_flops / 1000000)
    # s_total_params = 'Model PARAMs = {:.2f}M\n'.format(total_params / 1000000)
    # print(s_total_flops)
    # print(s_total_params)

    #### Get a list of several pruning rate through certain initialization methods
    # max_FLOPs_FP = 0.01
    max_FLOPs_FP = 0.1
    max_PARAMs_FP = 0.0 
    target_flops_fp = 0.18
    target_params_fp = 0.0 
    init_group_num = 80

    print('Setting global pruning rate: flops {:.2f}  |  params {:.2f}' .format(target_flops_fp, target_params_fp))
    print('Setting number of init group: ', init_group_num, end='\n\n')

    candidates_prune_fp = random_can_fp(model, max_FLOPs_FP, max_PARAMs_FP, target_params_fp, target_flops_fp, 
                imgsz, init_group_num, FP_Rate_Len, total_flops, total_params)
    
    # for candidator in candidates_prune_fp:
    #     for ca in candidator:
    #         print('%.4f' % ca, end='  ')
    #     print('\n')


    acc_total_candidates = []
    # loss_total_candidates = []
    cnt = 0
    for cur_pruning_rate in candidates_prune_fp:
        print('\n>>>>>> the {}th model'.format(cnt), flush=True)

        print('the current Filter Prune Rate : ')
        # for x in cur_pruning_rate[:-2]:
        #     print(" {:.3f} ".format(x), end=',')
        print(cur_pruning_rate[:-2])
        print('\nFLOPs = {:.2f}M ({:.2f}X | {:.2f}%) '.format(cur_pruning_rate[-1]/1000000, float(total_flops/cur_pruning_rate[-1]), float(1.0-cur_pruning_rate[-1]/total_flops)*100), flush=True)
        print('PARAMs = {:.2f}M ({:.2f}X | {:.2f}% ) '.format(cur_pruning_rate[-2]/1000000, float(total_params/cur_pruning_rate[-2]), float(1.0-cur_pruning_rate[-2]/total_params)*100), flush=True, end='\n\n')

        t_current_prune_fp = cur_pruning_rate[:-2]

        model, _ = train_func(hyp, tb_writer, opt, device, pruning_rate=t_current_prune_fp, 
                    nw_ref=0, warmup=True, savelog=False, setting_epochs=0, dataloader=dataloader, testloader=testloader, dataset=dataset)

        # s = ('\n\t\t' + '%10.10s' * 4 % ('loss_GIoU', 'loss_obj', 'loss_cls', 'total'))
        # print(s)
        # print('\t\t%10.4g' * 4 % (losses[0], losses[1], losses[2], losses[3]), end='\n\n')


        results, maps, times = test.test(opt.data,
                                         batch_size=total_batch_size,
                                         imgsz=imgsz_test,
                                        #  save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                         model=model,
                                         single_cls=opt.single_cls,
                                         dataloader=testloader)

        # losses = losses.cpu().numpy()

        # import sys
        # sys.exit()
        # print("\n")

        # print('\nScore = {:.4f}   {:.4f}\n'.format(float(results[2]), float(losses[3])), flush=True)
        print('\nScore = {:.4f}\n'.format(float(results[2])), flush=True)
        acc_total_candidates.append([cur_pruning_rate,float(results[2])])
        # loss_total_candidates.append([cur_pruning_rate, float(losses[3])])

        cnt = cnt + 1

    # import sys
    # sys.exit()


    # # pruning
    # acc_total_candidates = sorted(acc_total_candidates, key=lambda candidator: candidator[-1], reverse=True)
    # loss_total_candidates = sorted(loss_total_candidates, key=lambda candidator: candidator[-1], reverse=False)
    # # epoch_test = 1
    # print('----------------- result sorted by acc -----------------')
    # for candidator in acc_total_candidates:
    #     print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X | {:.2f}%) |'.format(candidator[0][-1]/1000000,float(total_flops/candidator[0][-1]),float(1.0-candidator[0][-1]/total_flops)*100), flush=True)
    #     print('PARAMs = {:.2f}M ({:.2f}X | {:.2f}% )|'.format(candidator[0][-2]/1000000,float(total_params/candidator[0][-2]),float(1.0-candidator[0][-2]/total_params)*100), flush=True)
    #     print('Score = {:.4f}   '.format(candidator[1]))
    # print('---------------------------------------------------------')

    # print('----------------- result sorted by loss -------------------')
    # for candidator in loss_total_candidates:
    #     print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X | {:.2f}%) |'.format(candidator[0][-1]/1000000,float(total_flops/candidator[0][-1]),float(1.0-candidator[0][-1]/total_flops)*100), flush=True)
    #     print('PARAMs = {:.2f}M ({:.2f}X | {:.2f}% )|'.format(candidator[0][-2]/1000000,float(total_params/candidator[0][-2]),float(1.0-candidator[0][-2]/total_params)*100), flush=True)
    #     print('Score = {:.4f}   '.format(candidator[1]))
    # print('------------------------------------------------------------')

    acc_candidates, acc_candidates_score = select_candidate(acc_total_candidates, 4, reverse=True)
    # loss_candidates, loss_candidates_score = select_candidate(loss_total_candidates, 4, reverse=False)

    print('\nmap candidates....')
    for acc_pruning_rate, score in zip(acc_candidates, acc_candidates_score):
        print('score: ', score)
        print(acc_pruning_rate)

    # print('\nloss candidates....')
    # for loss_pruning_rate, score in zip(loss_candidates, loss_candidates_score):
    #     print('score: ', score)
    #     print(loss_pruning_rate)


    search_acc_score = [score for score in acc_candidates_score]
    # search_loss_score = [score for score in loss_candidates_score]

    # search:map   train:map
    acc_train_acc_score = []
    # search:map   train:loss
    # loss_train_acc_score = [None] * len(loss_candidates)


    print('\n-----------------------------------------------------')
    print('Get train acc score by acc sorted....', end='\n\n')
    for acc_pruning_rate in acc_candidates:
        cur_pruning_rate = acc_pruning_rate[:-2]
        print(acc_pruning_rate)

        model, best_acc, _ = train_finalc(hyp, tb_writer, opt, device, pruning_rate=cur_pruning_rate,  
                        nw_ref=0, warmup=True, savelog=False, setting_epochs=10, dataloader=dataloader, testloader=testloader, dataset=dataset)

        
        # best_loss = best_loss.cpu().tolist()
        print('sorted by acc, current model best accuracy = {:.2f}    ' .format(float(best_acc)))
        # print('sorted by acc, current model best loss = {:.2f}      ' .format(float(best_loss)))

        acc_train_acc_score.append(best_acc)
        # for idx, loss_pruning_rate in enumerate(loss_candidates):
        #     if operator.eq(loss_pruning_rate[:-2], cur_pruning_rate):
        #         loss_train_acc_score[idx] = best_acc
        #         # train_loss_score[idx] = best_loss
        #         break
        #         # print(train_acc_loss)

    # print('\n---------------------------------------------------')
    # print('Get train acc score by loss sorted....', end='\n\n')
    # for idx, loss_pruning_rate in enumerate(loss_candidates):
    #     if loss_train_acc_score[idx] is None:
    #         cur_pruning_rate = loss_pruning_rate[:-2]

    #         model, best_acc, _ = train_finalc(hyp, tb_writer, opt, device, pruning_rate=cur_pruning_rate,
    #                         nw_ref=0, warmup=True, savelog=False, setting_epochs=20, dataloader=dataloader, testloader=testloader, dataset=dataset)

    #         # best_loss = best_loss.cpu().tolist()
    #         loss_train_acc_score[idx] = best_acc
    #         # print(train_acc_loss)

    #         print('sorted by loss, current model best accuracy = {:.2f}   ' .format(float(best_acc)))
    #         # print('sorted by loss, loss current model best loss = {:.2f}     ' .format(float(best_loss)))


    print('search acc score: \n', search_acc_score)
    # print('search loss score: \n', search_loss_score)
    print('search acc, train acc score: \n', acc_train_acc_score)
    # print('search loss, train acc score: \n', loss_train_acc_score)


    x_search_acc = pd.Series(search_acc_score)
    # x_search_loss = pd.Series(search_loss_score)

    y_acc_train_acc = pd.Series(acc_train_acc_score)
    # y_loss_train_acc = pd.Series(loss_train_acc_score)

    kendall_acc_acc = x_search_acc.corr(y_acc_train_acc, method='kendall')
    pearson_acc_acc = x_search_acc.corr(y_acc_train_acc, method='pearson')
    
    # kendall_loss_acc = x_search_loss.corr(y_loss_train_acc, method='kendall')
    # pearson_loss_acc = x_search_loss.corr(y_loss_train_acc, method='pearson')

    # kendall_acc_loss = x1.corr(y2, method='kendall')
    # pearson_acc_loss = x1.corr(y2, method='pearson')
    # kendall_loss_loss = x2.corr(y2, method='kendall')
    # pearson_loss_loss = x2.corr(y2, method='pearson')

    print('search by accuracy : test by accurary   kendall = {:.3f}     ' .format(float(kendall_acc_acc)))
    print('search by accuracy : test by accurary   pearson = {:.3f}     ' .format(float(pearson_acc_acc)))

    # print('search by loss : test by accurary   kendall = {:.3f}     ' .format(float(kendall_loss_acc)))
    # print('search by loss : test by accurary   pearson = {:.3f}     ' .format(float(pearson_loss_acc)))

    

    # print('accuracy : loss   kendall = {:.3f}     ' .format(float(kendall_acc_loss)))
    # print('accuracy : loss   pearson = {:.3f}     ' .format(float(pearson_acc_loss)))

    # print('loss : accurary   kendall = {:.3f}     ' .format(float(kendall_loss_loss)))
    # print('loss : accurary   pearson = {:.3f}     ' .format(float(pearson_loss_loss)))


    import sys
    sys.exit()

    model = model.to(device)
    


    ###############################
    #####   Start training   ######
    ###############################
    # Start training
    # nc: number of classes
    # nb: number of batches(iterations), num_training_pics / batch_size
    # nw: number of warmup iterations, max(3 epochs, 1k iterations)
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # print("number of warmup iterations: ", nw)
    # print("epochs: ", nb)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    if rank in [0, -1]:
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % epochs)
    
    ####################  epoch  ######################
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        # When in DDP mode, the generated indices will be broadcasted to synchronize dataset.
        # dataset.image_weights: default(false)
        if dataset.image_weights:
            print('Update image weights')
            # Generate indices.
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                 k=dataset.n)  # rand weighted idx
            # Broadcast.
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'lr'))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        ######################  batch ##########################
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # ni: idx of taining pics(global, 0-len(trainset))
            # nb: number of batches(iterations), num_training_pics / batch_size
            # nw: number of warmup iterations, max(3 epochs, 1k iterations)
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            ############   Warmup    ############
            # Warmup
            if ni <= nw and warmup:
                # print('---------warmup-----------')
                xi = [0, nw]  # x interp

                # y0 = np.interp(x0, xi, yi) 
                # xi: 横坐标， yi：纵坐标，(xi,yi) 获得一条 y-x 曲线，返回 x0 处的 y0 值
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                
                # print('\t\t\t\t epoch  ni/nw  nbs/total_batch_size ', epoch, ni, nw, nbs, total_batch_size)
                # print('xi ', xi)    xi  [0, 1000.0]
                # print('yi ', [1, nbs / total_batch_size])     yi  [1, 1.0]
                # print('accumulate ', accumulate)    accumulate  1

                # optimizer.param_groups：  pg0：all else     pg1: weights       pg2: biases
                # if: consic learning rate   每个image 有一个 x['lr']，x['momentum']
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

                    # print('\t\t\t\tj   lf(epoch)  x[initial_lr]  x[lr]  momentum   |  ', j, lf(epoch), x['initial_lr'], x['lr'], x['momentum'])


            ############  Multi-scale input ############
            # multi-scale: default(false)
            if opt.multi_scale:
                # print('Using multi scale')
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)


            #############  Forward  ##################
            # print('forward')
            pred = model(imgs)


            #############  Compute Loss   ################
            loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
            if rank != -1:
                loss *= opt.world_size  # gradient averaged between devices in DDP mode
            # 判断数据是否溢出
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            #############   Backward  #################
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            ############# Optimize   ##################
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
            
            if not warmup:
                # Scheduler
                scheduler.step()

            #############  Print    ###################
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6 + '   %s') % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1], optimizer.param_groups[0]['lr'])
                pbar.set_description(s)

                ### save train_batch.jpg
                # Plot 
                if ni < 3:
                    f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------
        ######################  batch ##########################

        if warmup:
            # Scheduler
            scheduler.step()


        ##################  test mAP  #####################
        # Only the first process in DDP mode is allowed to log or save checkpoints.
        if rank in [-1, 0]:
            # mAP
            if ema is not None:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir)

                # Write
                with open(results_file, 'a') as f:
                    f.write(s + '   %10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                if len(opt.name) and opt.bucket:
                    os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

                # Tensorboard
                if tb_writer:
                    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                            'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                    for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                        tb_writer.add_scalar(tag, x, epoch)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
                if fi > best_fitness:
                    best_fitness = fi

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if (best_fitness == fi) and not final_epoch:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload
        # Finish
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyp.yaml path (optional)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help="Total batch size for all gpus.")
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/to/last.pt, or most recent run if blank.')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_args()


    # # Configure
    # with open(opt.data) as f:
    #     data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    # train_path = data_dict['train']
    # test_path = data_dict['val']
    # nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    # assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # # Create model
    # # model = Model(opt.cfg, nc=nc).to(device)
    # model = Model(opt.cfg, nc=nc)
    # dummy_input = torch.randn(1, 3, 640, 640)
    # torch.onnx.export(model, 
    #                 dummy_input,
    #                 "test_640.onnx",
    #                 verbose = True,
    #                 opset_version = 9,
    #                 input_names = ['images'],
    #                 do_constant_folding=True)
    #                 #output_names = ['output', '365', '379'])




    # from torchsummary import summary
    # summary(model, (3, 320, 320))

    # # #################################################
    # # # Torch Pruning (Begin)
    # # #################################################
    # import sys
    # import torch_pruning as tp
    # # import dependency
    # model.eval()
    # num_params_before_pruning = tp.utils.count_params( model )
    # print(num_params_before_pruning)

    # # 1. build dependency graph
    # strategy = tp.strategy.L1Strategy()
    # DG = tp.DependencyGraph()
    # out = model(torch.randn([1,3, opt.img_size[0], opt.img_size[0]]))
    # DG.build_dependency(model, example_inputs=torch.randn([1,3, opt.img_size[0], opt.img_size[0]]))
    # excluded_layers = list(model.model[-1].modules())
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d) and m not in excluded_layers:
    #         pruning_plan = DG.get_pruning_plan( m, tp.prune_conv, idxs=strategy(m.weight, amount=0.4) )
    #         print("pruning plan ")
    #         print(pruning_plan)
    #         # execute the plan (prune the model)
    #         pruning_plan.exec()
    # num_params_after_pruning = tp.utils.count_params( model )
    # print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))
    # #################################################
    # # Torch Pruning (End)
    # #################################################

    # device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    # opt.total_batch_size = opt.batch_size
    # opt.world_size = 1
    # if device.type == 'cpu':
    #     mixed_precision = False
    # elif opt.local_rank != -1:
    #     # DDP mode
    #     assert torch.cuda.device_count() > opt.local_rank
    #     torch.cuda.set_device(opt.local_rank)
    #     device = torch.device("cuda", opt.local_rank)
    #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

    #     opt.world_size = dist.get_world_size()
    #     assert opt.batch_size % opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
    #     opt.batch_size = opt.total_batch_size // opt.world_size
    # print(opt)
    # model = model.to(device)



    ##########################
    ###### load weights ###### 
    ##########################
    # resume: default(False) 
    # if its true, load weights from latest run
    last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # resume from most recent run
    if last and not opt.weights:
        print(f'Resuming training from {last}')
    opt.weights = last if opt.resume and not opt.weights else opt.weights
    
    ##########################################
    ####### check file/data,model, hyp  ###### 
    ##########################################
    # hyp: default=''
    # cfg: model
    # data: dataset
    # if opt.local_rank in [-1, 0]:
    #     check_git_status()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    if opt.hyp:  # update hyps
        opt.hyp = check_file(opt.hyp)  # check file
        with open(opt.hyp) as f:
            hyp.update(yaml.load(f, Loader=yaml.FullLoader))  # update hyps
    
    ###########################
    ### init some opt.param ### 
    ###########################
    # mixed_precision: true
    # local_rank: -1
    # DDP mode: distributed data parallel training
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    if device.type == 'cpu':
        mixed_precision = False
    elif opt.local_rank != -1:
        # DDP mode
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

        opt.world_size = dist.get_world_size()
        assert opt.batch_size % opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
        opt.batch_size = opt.total_batch_size // opt.world_size
    print(opt)

    ###########################
    ###       Train         ### 
    ###########################
    # evolve: default=false
    if not opt.evolve:
        if opt.local_rank in [-1, 0]:
            if (savelog):
                print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
                tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', opt.name))
            else:
                tb_writer = None
        else:
            tb_writer = None
        train(hyp, tb_writer, opt, device)

    # 超参数进化(遗传算法)  https://zhuanlan.zhihu.com/p/123319468
    # Evolve hyperparameters (optional)
    else:
        assert opt.local_rank == -1, "DDP mode currently not implemented for Evolve!"

        tb_writer = None
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(10):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(hyp.copy(), tb_writer, opt, device)

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)



# python train_pruning.py --cfg ./models/yolov5s_ori.yaml --data data/quexian.yaml --img-size 320 --weights weights/DET-biaopan_quexian-20211028.pt --batch-size 32 --epochs 10 --device 0