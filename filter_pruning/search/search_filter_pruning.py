import os
import math
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import copy


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_pruning as tp
from filter_pruning.torch_pruning.pruning import filter_pruning_random
from fp_utils.utils import *



def softmax_numpy(x):
    e_x = np.exp(x.astype(float) - np.max(x))
    return e_x / e_x.sum()


def uniform_normal_init(loc, scale, size): 
    a = np.random.normal(loc=loc,scale=scale,size=size)
    mask = a <= loc
    b = np.random.uniform(0.001,loc,size)
    out = a*(1-mask)+b*mask
    return out

def mix_normal_init(loc, size, max_value):
    out = []
    scale1 = (max_value-loc)/3.0
    scale2 = (loc-0.001)/3.0
    cnt = 0
    while cnt < size:
        a = np.random.normal(loc=loc,scale=scale1)
        if a > 0.1:
            a = max_value if (a > max_value) else a
            out.append(a)
        else:
            while True:
                b=np.random.normal(loc=loc,scale=scale2)
                if b < 0.1 and b>0:
                    out.append(b)
                    break
        cnt+=1
    out = np.array(out)
    return out

def saturation_value(a, min, max):
    m1 = a < min
    # a = (0.5/args.max_PARAMs)*m1+a*(1-m1)
    a = min*m1+a*(1-m1)
    m2 = a > max
    a = max*m2+a*(1-m2)
    return a


def random_can_fp(model, max_FLOPs_FP, max_PARAMs_FP, target_params_fp, target_flops_fp, 
            input_size, num, num_states, total_flops, total_params):
    print('random select ........', flush=True)
    candidates = []
    candidates_int = []
    # loop_num = 0
    while len(candidates)<num:
        if max_FLOPs_FP == 0 and max_PARAMs_FP == 0:
            can = np.zeros(num_states+2)
            can[-1] = total_flops
            can[-2] = total_params
            print('pruning rate = 0')          
        else:
            # if loop_num % 20==0:
            #     print('loop start: %d' % loop_num)
            # if(loop_num > 200):
            #     break          
            # loop_num += 1
            # # uniform random init
            # high = 2.0/max(max_PARAMs_FP,max_FLOPs_FP)
            # can = np.random.uniform(0.01,high,num_states+2)

            # ## uniform normal init
            # # mu = 1/max(max_PARAMs_FP,max_FLOPs_FP)
            # mu = max(max_PARAMs_FP,max_FLOPs_FP)
            # sigma = (0.5-mu) / 3.0
            # can = uniform_normal_init(mu, sigma, num_states+2)

            # # # mix normal_init
            # loc = 1.0/max(max_PARAMs_FP,max_FLOPs_FP)
            # can = mix_normal_init(loc,num_states+2,0.99)

            # # gamma init
            loc = max(max_PARAMs_FP, max_FLOPs_FP)
            # can = np.random.gamma(loc*8.0, 0.125, num_states + 2)
            can = np.random.gamma(loc*8.0, 0.125, num_states + 2)
            mask1 = can >= 0.9999
            mask2 = can <= 0.0001
            can = mask1 * 0.9999 + (1 - mask1) * can
            can = mask2 * 0.0001 + (1 - mask2) * can

            t_can = tuple(can[:-2])
            # print(can[:-2].tolist())

            # model = models.__dict__[args.arch]()


            fp_model = filter_pruning_random(model,torch.randn(1,3,input_size,input_size),output_transform=None,pruning_rates=t_can)

            layer_flops = calc_model_flops(fp_model, input_size, mul_add=False)
            sparse_total_flops = sum(layer_flops)
            layer_params = calc_model_parameters(fp_model)
            sparse_total_params = sum(layer_params)
            # print('pruning rate (flops): ', 1.0-sparse_total_flops/total_flops)
            # print('pruning rate (params): ', 1.0-sparse_total_params/total_params)

            if max_FLOPs_FP == 0 and max_PARAMs_FP !=0:
                if 1.0-sparse_total_params/total_params < target_params_fp or 1.0-sparse_total_params/total_params > target_params_fp+0.1:
                    continue
            elif max_FLOPs_FP != 0 and max_PARAMs_FP ==0:
                if 1.0-sparse_total_flops/total_flops < target_flops_fp or 1.0-sparse_total_flops/total_flops > target_flops_fp+0.1:
                    continue
            else:
                if 1.0-sparse_total_flops/total_flops < target_flops_fp or 1.0-sparse_total_flops/total_flops > target_flops_fp+0.1 or 1.0-sparse_total_params/total_params < target_params_fp or 1.0-sparse_total_params/total_params > target_params_fp+0.1:
                    continue

            can[-1] = sparse_total_flops
            can[-2] = sparse_total_params

            # compare difference
            t_can_int = [math.ceil(i*100.0) for i in t_can]


            if t_can_int in candidates_int:
                continue
            else:
                candidates_int.append(t_can_int)

        # print(total_flops/sparse_total_flops)
        print('number of candidates: ', len(candidates))
        candidates.append(can.tolist())

        # print(len(candidates))
        # print(len(candidates_int))

    # print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates

# mutation operation in evolution algorithm
def get_mutation_fp(model, max_FLOPs_FP, max_PARAMs_FP, target_params_fp, target_flops_fp, 
        input_size, epoch, keep_top_candidates, top_candidates_score, num_states, mutation_num, 
        m_prob, strength, total_flops, total_params):
    
    print('mutation ......', flush=True)
    res = []
    candidates_int = []
    global_candidates_int = []

    for candidator in keep_top_candidates:
        global_candidates_int.append([math.ceil(i*100.0) for i in candidator[:-2]])
    
    k = len(keep_top_candidates)
    iter = 0
    max_iters = 10*mutation_num
    # top_candidates_score = top_candidates_score / 100.0
    top_k_p = softmax_numpy(top_candidates_score)
    while len(res)<mutation_num and iter<max_iters:
    # while len(res)<mutation_num:

        if max_FLOPs_FP == 0 and max_PARAMs_FP == 0:
            can = np.zeros(num_states+2)
            can[-1] = total_flops
            can[-2] = total_params     
            res.append(can.tolist())       
        else:
            # ids = np.random.choice(k, mutation_num,replace=False,p=top_k_p)
            # ids = np.random.choice(k, mutation_num,replace=False)
            # ids = np.random.choice(k, mutation_num,p=top_k_p)
            ids = np.random.choice(k, mutation_num)
            # print(ids)
            select_seed = np.array([keep_top_candidates[id] for id in ids])
            # if epoch < 50:
            #     alpha = np.random.uniform(0,1,num_states+2)
            # elif epoch < 100:
            #     alpha = np.random.uniform(0.25,0.75,num_states+2)
            # elif epoch < 120:
            #     alpha = np.random.uniform(0.35,0.65,num_states+2)
            # alpha = np.random.uniform(0,1,num_states+2)
            alpha = np.random.normal(loc=0.5,scale=0.1,size=num_states+2)
            is_m = np.random.choice(np.arange(0,2), (mutation_num, num_states+2), p=[1-m_prob, m_prob]).astype(np.float32)
            # is_m[:,0] = np.random.choice(np.arange(0,2),len(is_m[:,0]) , p=[0.6, 0.4]).astype(np.float32)
            # is_m[:,-1] = np.random.choice(np.arange(0,2),len(is_m[:,-1]) , p=[0.6, 0.4]).astype(np.float32)
            # select_list =  select_seed*(1.0-is_m)+(1.0-select_seed)*is_m
            mask1 = alpha >= 1
            mask2 = alpha <= 0
            alpha = alpha*(1-mask1)+0.5*(mask1) 
            alpha = alpha*(1-mask2)+0.5*(mask2)
            mask = alpha < 0.5
            beta = (pow((2*alpha),(1/strength))-1)*mask + (1-pow((2*(1-alpha)),(1/strength)))*(1-mask)
            select_list =  select_seed+beta*is_m
            select_list = saturation_value(select_list, 0.0001, 0.9999)
            iter += 1
            cnt = 0
            for can in select_list:
                cnt = cnt + 1
                sum_mask = sum(is_m[cnt-1])
                t_can = tuple(can[:-2])

                # model = models.__dict__[args.arch]()

                fp_model = filter_pruning_random(model,torch.randn(1,3,input_size,input_size), output_transform=None, pruning_rates=t_can)

                layer_flops = calc_model_flops(fp_model,input_size,mul_add=False)
                sparse_total_flops = sum(layer_flops)
                layer_params = calc_model_parameters(fp_model)
                sparse_total_params = sum(layer_params)

                if max_FLOPs_FP == 0 and max_PARAMs_FP !=0:
                    if 1.0-sparse_total_params/total_params < target_params_fp or 1.0-sparse_total_params/total_params > target_params_fp+0.1:
                        continue
                elif max_FLOPs_FP != 0 and max_PARAMs_FP ==0:
                    if 1.0-sparse_total_flops/total_flops < target_flops_fp or 1.0-sparse_total_flops/total_flops > target_flops_fp+0.1:
                        continue
                else:
                    if 1.0-sparse_total_flops/total_flops < target_flops_fp or 1.0-sparse_total_flops/total_flops > target_flops_fp+0.1 \
                    or 1.0-sparse_total_params/total_params < target_params_fp or 1.0-sparse_total_params/total_params > target_params_fp+0.1:
                        continue
                can[-1] = sparse_total_flops
                can[-2] = sparse_total_params
                # print(total_flops/sparse_total_flops)
                # print(total_params/sparse_total_params)

                # compare difference
                t_can_int = [math.ceil(i*100.0) for i in t_can]
                if (t_can_int in candidates_int) or (t_can_int in global_candidates_int):
                    continue
                else:
                    candidates_int.append(t_can_int)

                res.append(can)
                if len(res)==mutation_num:
                    break
    print('mutation_num = {}'.format(len(res)), flush=True)
    return res

# crossover operation in evolution algorithm
def get_crossover_fp(model, max_FLOPs_FP, max_PARAMs_FP, target_params_fp, target_flops_fp, 
        input_size, keep_top_candidates, top_candidates_score, num_states, crossover_num, total_flops,total_params):
    
    print('crossover ......', flush=True)
    res = []
    candidates_int = []
    global_candidates_int = []
    for candidator in keep_top_candidates:
        global_candidates_int.append([math.ceil(i*100.0) for i in candidator[:-2]])
    
    k = len(keep_top_candidates)
    # top_candidates_score = top_candidates_score / 100.0
    top_k_p = softmax_numpy(top_candidates_score)
    iter = 0
    max_iters = 10 * crossover_num
    while len(res)<crossover_num and iter<max_iters:
    # while len(res)<crossover_num:

        if max_FLOPs_FP == 1 and max_PARAMs_FP == 1:
            can = np.zeros(num_states+2)
            can[-1] = total_flops
            can[-2] = total_params     
            res.append(can.tolist())       
        else:
            # id1, id2 = np.random.choice(k, 2, replace=False)
            # id1, id2 = np.random.choice(k, 2, replace=False,p=top_k_p)
            # print([id1,id2])
            # p1 = keep_top_candidates[id1]
            # p2 = keep_top_candidates[id2]
            id1 = np.random.choice(k, 1, p=top_k_p)
            id2 = np.random.choice(k, 1)
            # print([id1[0],id2[0]])
            p1 = keep_top_candidates[id1[0]]
            p2 = keep_top_candidates[id2[0]]

            # recombination
            alpha = np.random.rand(len(p1))
            can = p1*alpha + p2*(1.0-alpha)

            # alpha = np.random.uniform(-1, 1, len(p1))
            # can = p1+alpha*(p1-p2)

            can = saturation_value(can, 0.0001, 0.9999)
            # ## Discrete recombination
            # mask = np.random.randint(low=0, high=2, size=(num_states+2)).astype(np.float32)
            # can = p1*mask + p2*(1.0-mask)
            iter += 1
            t_can = tuple(can[:-2])

            # model = models.__dict__[args.arch]()

            fp_model = filter_pruning_random(model, torch.randn(1,3,input_size,input_size),output_transform=None,pruning_rates=t_can)

            layer_flops = calc_model_flops(fp_model,input_size,mul_add=False)
            sparse_total_flops = sum(layer_flops)
            layer_params = calc_model_parameters(fp_model)
            sparse_total_params = sum(layer_params)

            if max_FLOPs_FP == 0 and max_PARAMs_FP !=0:
                if 1.0-sparse_total_params/total_params < target_params_fp or 1.0-sparse_total_params/total_params > target_params_fp+0.1:
                    continue
            elif max_FLOPs_FP != 0 and max_PARAMs_FP ==0:
                if 1.0-sparse_total_flops/total_flops < target_flops_fp or 1.0-sparse_total_flops/total_flops > target_flops_fp+0.1:
                    continue
            else:
                if 1.0-sparse_total_flops/total_flops < target_flops_fp or 1.0-sparse_total_flops/total_flops > target_flops_fp+0.1 or \
                    1.0-sparse_total_params/total_params < target_params_fp or 1.0-sparse_total_params/total_params > target_params_fp+0.1:
                    continue

            can[-1] = sparse_total_flops
            can[-2] = sparse_total_params
            # print(total_flops/sparse_total_flops)
            # print(total_params/sparse_total_params)

            # compare difference
            t_can_int = [math.ceil(i*100.0) for i in t_can]
            if (t_can_int in candidates_int) or (t_can_int in global_candidates_int):
                continue
            else:
                candidates_int.append(t_can_int)

            res.append(can)
            if len(res)==crossover_num:
                break
    print('crossover_num = {}'.format(len(res)), flush=True)
    return res
