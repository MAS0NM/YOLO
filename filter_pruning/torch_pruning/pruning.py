import torch
import torch.nn as nn

import random
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_pruning as tp
from fp_utils.utils import *


## 1. deep copy from input model
## 2. add pruning rate for each layers
## 3. return pruned model
def filter_pruning_random(model_ref, example_inputs, output_transform, pruning_rates):
    model=copy.deepcopy(model_ref)
    model.cpu().eval()

    # layer_flops = calc_model_flops(model, 320, mul_add=False)
    # sparse_total_flops = sum(layer_flops)
    # layer_params = calc_model_parameters(model)
    # sparse_total_params = sum(layer_params)
    # print('before pruning (flops): ', sparse_total_flops)
    # print('before pruning (params): ', sparse_total_params)

    prunable_module_type = (nn.Conv2d)
    prunable_modules_all = [ m for m in model.modules() if isinstance(m, prunable_module_type) ]
    prunable_modules = prunable_modules_all[:len(prunable_modules_all)-3].copy()
    # print('number of prunable modules: ', len(prunable_modules))
    DG = tp.DependencyGraph()
    DG.build_dependency( model, example_inputs=example_inputs, output_transform=output_transform )

    for layer_to_prune, fp_rate in zip(prunable_modules, pruning_rates):
        if isinstance( layer_to_prune, nn.Conv2d ):
            prune_fn = tp.prune_conv
        weight = layer_to_prune.weight.detach().cpu().numpy()
        out_channels = weight.shape[0]
        num_pruned = int(out_channels * fp_rate)
        rand_idx = random.sample( list(range(out_channels)),  num_pruned )
        plan = DG.get_pruning_plan( layer_to_prune, prune_fn, rand_idx)
        plan.exec()

    # layer_flops = calc_model_flops(model, 320, mul_add=False)
    # sparse_total_flops = sum(layer_flops)
    # layer_params = calc_model_parameters(model)
    # sparse_total_params = sum(layer_params)
    # print('after pruning (flops): ', sparse_total_flops)
    # print('after pruning (params): ', sparse_total_params)

    # with torch.no_grad():
    #     out = model( example_inputs )
    #     if output_transform:
    #         out = output_transform(out)
    return model


def filter_pruning(model_ref, imgsz, pruning_rate, pruning_strategy=None, savelog=False, savefile=''):
    model = copy.deepcopy(model_ref) 
    
    # layer_params = calc_model_parameters(model)
    # layer_flops = calc_model_flops(model, imgsz, mul_add=False)
    # before_params = sum(layer_params)
    # before_flops = sum(layer_flops)

    model.eval()
    # num_params_before_pruning = tp.utils.count_params( self.model )
    # 1. build dependency graph
    strategy = tp.strategy.L1Strategy()
    # strategy = tp.strategy.L2Strategy()
    # strategy = tp.strategy.RandomStrategy()
    DG = tp.DependencyGraph()
    # out = model(torch.randn([1,3, opt.img_size[0], opt.img_size[0]]))
    DG.build_dependency(model, example_inputs=torch.randn([1,3, imgsz, imgsz]))
        
    pruning_idx = 0
    excluded_layers = list(model.model[-1].modules())
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m not in excluded_layers:
            # pruning_plan = DG.get_pruning_plan( m, tp.prune_conv, idxs=strategy(m.weight, amount=0.0) )
            pruning_plan = DG.get_pruning_plan( m, tp.prune_conv, idxs=strategy(m.weight, amount=pruning_rate[pruning_idx]) )
            pruning_idx += 1
            # print(pruning_plan)
            if savelog:
                with open(savefile, 'a') as file:
                    pruning_plan.write_file(file)
            # execute the plan (prune the model)
            pruning_plan.exec()

    # layer_params = calc_model_parameters(model)
    # layer_flops = calc_model_flops(model, imgsz, mul_add=False)
    # after_params = sum(layer_params)
    # after_flops = sum(layer_flops)
    # pruning_flops = '\nFilter Pruning FLOPs = {:.2f}M | {:.2f}M | ({:.2f}X | {:.2f}%) '.format(
    #                     before_flops / 1000000, 
    #                     after_flops/1000000, 
    #                     float(before_flops/after_flops), 
    #                     float(1.0-after_flops/before_flops)*100)

    # pruning_param = 'Filter Pruning PARAMs = {:.2f}M | {:.2f}M | ({:.2f}X | {:.2f}%) \n'.format(
    #                     before_params / 1000000, 
    #                     after_params/1000000, 
    #                     float(before_params/after_params), 
    #                     float(1.0-after_params/before_params)*100, end='\n\n')
    # print(pruning_param)
    # print(pruning_flops)
    return model