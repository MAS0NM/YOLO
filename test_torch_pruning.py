'''
Copyright (c) 2021 Leinao
All rights reserved.

Author: username
Date: 2022-02-15 16:07:06
LastEditors: username
LastEditTime: 2022-02-15 16:43:24
'''


import argparse

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


    # Configure
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Create model
    model = Model(opt.cfg, nc=nc)
    # from torchsummary import summary
    # summary(model, (3, 320, 320))

    # #################################################
    # # Torch Pruning (Begin)
    # #################################################
    import sys
    import torch_pruning as tp
    # import dependency
    model.eval()
    num_params_before_pruning = tp.utils.count_params( model )
    print(num_params_before_pruning)

    # 1. build dependency graph
    strategy = tp.strategy.L1Strategy()
    DG = tp.DependencyGraph()
    out = model(torch.randn([1,3, opt.img_size[0], opt.img_size[0]]))
    DG.build_dependency(model, example_inputs=torch.randn([1,3, opt.img_size[0], opt.img_size[0]]))
    excluded_layers = list(model.model[-1].modules())
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m not in excluded_layers:
            pruning_plan = DG.get_pruning_plan( m, tp.prune_conv, idxs=strategy(m.weight, amount=0) )
            print("pruning plan ")
            print(pruning_plan)
            # execute the plan (prune the model)
            pruning_plan.exec()
    num_params_after_pruning = tp.utils.count_params( model )
    print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))
    #################################################
    # Torch Pruning (End)
    #################################################

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