import argparse
from audioop import reverse
from crypt import methods

import torch.distributed as dist
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import pandas as pd
import operator
import numpy as np
import sys


####### YOLO ########
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *
from utils.read_write import write_pruning_rate

####### Pruning ########
from filter_pruning.search.search_filter_pruning import *
from filter_pruning.fp_utils.utils import *
import filter_pruning.torch_pruning as tp
# from train_fun import train_func
# from train_final import train_finalc
from filter_pruning.fp_train import Training
from filter_pruning.torch_pruning.pruning import filter_pruning
import filter_pruning.fp_test as test


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



def search_filter_pruning_correlation(hyp, tb_writer, opt, device):

    print(f'Hyperparameters {hyp}')

    trainer = Training(hyp, opt, device, mixed_precision=mixed_precision, savelog=False) 

    trainer.create_optimizer()
    trainer.load_model_weights()

    # epochs
    trainer.create_scheduler(warmup=True)

    model_init, imgsz = trainer.get_model()

    res = trainer.test(in_model=model_init)
    print(res[2])

    sys.exit()

    #################################################
    # Torch Pruning (Begin)
    #################################################

    print('start pruning.....')

    # prepare candidates
    prunable_module_type = (nn.Conv2d)
    prunable_modules = [m for m in model_init.modules() if isinstance(m, prunable_module_type)]
    FP_Rate_Len = len(prunable_modules)
    FP_Rate_Len -= 3
    print('Number of prunable convolution ', FP_Rate_Len)


    layer_params = calc_model_parameters(model_init)
    layer_flops = calc_model_flops(model_init, input_size=imgsz, mul_add=False)
    total_params = sum(layer_params)
    total_flops = sum(layer_flops)
    # s_total_flops = '\nModel FLOPs = {:.2f}M'.format(total_flops / 1000000)
    # s_total_params = 'Model PARAMs = {:.2f}M\n'.format(total_params / 1000000)
    # print(s_total_flops)
    # print(s_total_params)
    print(total_params)
    print(total_flops)


    #### Get a list of several pruning rate through certain initialization methods
    # max_FLOPs_FP = 0.01
    max_FLOPs_FP = 0.1
    max_PARAMs_FP = 0.0 
    target_flops_fp = 0.18
    target_params_fp = 0.0 
    init_group_num = 2

    print('Setting global pruning rate: flops {:.2f}  |  params {:.2f}' .format(target_flops_fp, target_params_fp))
    print('Setting number of init group: ', init_group_num, end='\n\n')

    candidates_prune_fp = random_can_fp(model_init, max_FLOPs_FP, max_PARAMs_FP, target_params_fp, target_flops_fp, 
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
        print(cur_pruning_rate[:-2])
        print('\nFLOPs = {:.2f}M ({:.2f}X | {:.2f}%) '.format(cur_pruning_rate[-1]/1000000, float(total_flops/cur_pruning_rate[-1]), float(1.0-cur_pruning_rate[-1]/total_flops)*100), flush=True)
        print('PARAMs = {:.2f}M ({:.2f}X | {:.2f}% ) '.format(cur_pruning_rate[-2]/1000000, float(total_params/cur_pruning_rate[-2]), float(1.0-cur_pruning_rate[-2]/total_params)*100), flush=True, end='\n\n')

        t_current_prune_fp = cur_pruning_rate[:-2]

        model = filter_pruning(model_init, imgsz, t_current_prune_fp)
        model = model.to(device)

        results = trainer.test(in_model=model)

        # model = trainer.train(setting_epochs=1, in_nw=0, in_model=model, warmup=True)
        layer_params = calc_model_parameters(model)
        layer_flops = calc_model_flops(model, input_size=imgsz, mul_add=False)
        total_params = sum(layer_params)
        total_flops = sum(layer_flops)
        # print('after pruning.......')
        # s_total_flops = '\nModel FLOPs = {:.2f}M'.format(total_flops / 1000000)
        # s_total_params = 'Model PARAMs = {:.2f}M\n'.format(total_params / 1000000)
        # print(s_total_flops)
        # print(s_total_params)
        print(total_params)
        print(total_flops)

        model = trainer.adaptive_BN(in_model=model)

        layer_params = calc_model_parameters(model)
        layer_flops = calc_model_flops(model, input_size=imgsz, mul_add=False)
        total_params = sum(layer_params)
        total_flops = sum(layer_flops)
        # print('after ada BN.......')
        # s_total_flops = '\nModel FLOPs = {:.2f}M'.format(total_flops / 1000000)
        # s_total_params = 'Model PARAMs = {:.2f}M\n'.format(total_params / 1000000)
        # print(s_total_flops)
        # print(s_total_params)
        print(total_params)
        print(total_flops)

        results = trainer.test(in_model=model)

        layer_params = calc_model_parameters(model)
        layer_flops = calc_model_flops(model, input_size=imgsz, mul_add=False)
        total_params = sum(layer_params)
        total_flops = sum(layer_flops)
        # print('after test .......')
        # s_total_flops = '\nModel FLOPs = {:.2f}M'.format(total_flops / 1000000)
        # s_total_params = 'Model PARAMs = {:.2f}M\n'.format(total_params / 1000000)
        # print(s_total_flops)
        # print(s_total_params)
        print(total_params)
        print(total_flops)

        # # losses = losses.cpu().numpy()
        # print('\nScore = {:.4f}   {:.4f}\n'.format(float(results[2]), float(losses[3])), flush=True)
        # loss_total_candidates.append([cur_pruning_rate, float(losses[3])])

        # print('\nScore = {:.4f}\n'.format(float(results[2])), flush=True)
        print('\nScore', results[2])
        acc_total_candidates.append([cur_pruning_rate,float(results[2])])

        cnt = cnt + 1

    # import sys
    # sys.exit()


    # pruning
    acc_total_candidates = sorted(acc_total_candidates, key=lambda candidator: candidator[-1], reverse=True)
    # loss_total_candidates = sorted(loss_total_candidates, key=lambda candidator: candidator[-1], reverse=False)
    
    print('----------------- result sorted by acc -----------------')
    for candidator in acc_total_candidates:
        print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X | {:.2f}%) |'.format(candidator[0][-1]/1000000,float(total_flops/candidator[0][-1]),float(1.0-candidator[0][-1]/total_flops)*100), flush=True)
        print('PARAMs = {:.2f}M ({:.2f}X | {:.2f}% )|'.format(candidator[0][-2]/1000000,float(total_params/candidator[0][-2]),float(1.0-candidator[0][-2]/total_params)*100), flush=True)
        print('Score = {:.4f}   '.format(candidator[1]))
    print('---------------------------------------------------------')

    # print('----------------- result sorted by loss -------------------')
    # for candidator in loss_total_candidates:
    #     print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X | {:.2f}%) |'.format(candidator[0][-1]/1000000,float(total_flops/candidator[0][-1]),float(1.0-candidator[0][-1]/total_flops)*100), flush=True)
    #     print('PARAMs = {:.2f}M ({:.2f}X | {:.2f}% )|'.format(candidator[0][-2]/1000000,float(total_params/candidator[0][-2]),float(1.0-candidator[0][-2]/total_params)*100), flush=True)
    #     print('Score = {:.4f}   '.format(candidator[1]))
    # print('------------------------------------------------------------')

    acc_candidates, acc_candidates_score = select_candidate(acc_total_candidates, 1, reverse=True)
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

        # model, best_acc, _ = train_finalc(hyp, tb_writer, opt, device, pruning_rate=cur_pruning_rate,  
        #                 nw_ref=0, warmup=True, savelog=False, setting_epochs=1, dataloader=dataloader, testloader=testloader, dataset=dataset)

        layer_params = calc_model_parameters(model_init)
        layer_flops = calc_model_flops(model_init, input_size=imgsz, mul_add=False)
        total_params = sum(layer_params)
        total_flops = sum(layer_flops)
        # print('after test .......')
        # s_total_flops = '\nModel FLOPs = {:.2f}M'.format(total_flops / 1000000)
        # s_total_params = 'Model PARAMs = {:.2f}M\n'.format(total_params / 1000000)
        # print(s_total_flops)
        # print(s_total_params)
        print(total_params)
        print(total_flops)
        
        model = filter_pruning(model_init, imgsz, cur_pruning_rate)
        model = model.to(device)

        layer_params = calc_model_parameters(model)
        layer_flops = calc_model_flops(model, input_size=imgsz, mul_add=False)
        total_params = sum(layer_params)
        total_flops = sum(layer_flops)
        # print('after test .......')
        # s_total_flops = '\nModel FLOPs = {:.2f}M'.format(total_flops / 1000000)
        # s_total_params = 'Model PARAMs = {:.2f}M\n'.format(total_params / 1000000)
        # print(s_total_flops)
        # print(s_total_params)
        print(total_params)
        print(total_flops)

        results = trainer.test(in_model=model)

        # model = trainer.adaptive_BN(in_model=model)
        model, best_acc = trainer.train(setting_epochs=1, in_nw=50, in_model=model, warmup=True)
        # results = trainer.test(in_model=model)

        layer_params = calc_model_parameters(model)
        layer_flops = calc_model_flops(model, input_size=imgsz, mul_add=False)
        total_params = sum(layer_params)
        total_flops = sum(layer_flops)
        # print('after test .......')
        # s_total_flops = '\nModel FLOPs = {:.2f}M'.format(total_flops / 1000000)
        # s_total_params = 'Model PARAMs = {:.2f}M\n'.format(total_params / 1000000)
        # print(s_total_flops)
        # print(s_total_params)
        print(total_params)
        print(total_flops)

        
        # best_loss = best_loss.cpu().tolist()
        # print('sorted by acc, current model best accuracy = {:f}    ' .format(float(best_acc)))
        print(best_acc)
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

    print('search by accuracy : test by accurary   kendall = {:.3f}     ' .format(float(kendall_acc_acc)))
    print('search by accuracy : test by accurary   pearson = {:.3f}     ' .format(float(pearson_acc_acc)))

    # print('search by loss : test by accurary   kendall = {:.3f}     ' .format(float(kendall_loss_acc)))
    # print('search by loss : test by accurary   pearson = {:.3f}     ' .format(float(pearson_loss_acc)))


    import sys
    sys.exit()

    


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
    if opt.local_rank in [-1, 0]:
        if (savelog):
            print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', opt.name))
        else:
            tb_writer = None
    else:
        tb_writer = None
    search_filter_pruning_correlation(hyp, tb_writer, opt, device)




# python train_pruning.py --cfg ./models/yolov5s_ori.yaml --data data/quexian.yaml --img-size 320 --weights weights/DET-biaopan_quexian-20211028.pt --batch-size 32 --epochs 10 --device 0