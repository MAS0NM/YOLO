'''
Copyright (c) 2021 Leinao
All rights reserved.

Author: username
Date: 2022-02-18 10:03:28
LastEditors: username
LastEditTime: 2022-02-18 15:39:11
'''
from numpy import indices
import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random
import warnings
import numpy as np
from scipy.spatial import distance

# https://github.com/VainF/Torch-Pruning/issues/49 by @Serjio42
def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """
    n_remain = round_to*max(int(total_parameters - n_to_prune)//round_to, 1)
    return max(total_parameters - n_remain, 0)

class BaseStrategy(ABC):
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        # """ Apply the strategy on weights with user specified pruning percentage.

        # Parameters:
        #     weights (torch.Parameter): weights to be pruned.
        #     amount (Callable): the percentage of weights to be pruned (amount<1.0) or the amount of weights to be pruned (amount>=1.0) 
        #     round_to (int): the number to which the number of pruned channels is rounded.
        # """
        raise NotImplementedError
    
class GeometricMedian(BaseStrategy):
    
    def __init__(self, model):
        self.model_size = {}    #每一层的4维卷积参数形状
        self.model_length = {}  #每一层的参数vector长度
        self.compress_rate = {}
        self.distance_rate = {}
        self.model = model
        self.mask_index = []
        self.filter_large_index = {}
        self.similar_matrix = {}
    
    def get_filter_similar(self, weights, compress_rate, distance_rate, length, mix=False, dist_type="l2"):
        '''
            weights: torch.nn.parameter.Parameter
            length: model_length[index]
            weights.size(): out_channel, in_channel, w, h
        '''
        codebook = np.ones(length)
        if len(weights.size()) == 4:
            filter_pruned_num = int(weights.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weights.size()[0] * distance_rate)
            weight_vec = weights.view(weights.size()[0], -1)
            '''把该层的每个filter拉成一整条vector'''
            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_large_index = []
            
            if mix:
                filter_large_index = norm_np.argsort()[filter_pruned_num:]
            else:
                filter_large_index = norm_np.argsort()[:]

            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            '''选取weight_vec第0维的指定index的张量'''
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
                '''计算两个集合中每一对之间的距离'''
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            print('filter_large_index', filter_large_index)
            print('similar_sum', similar_sum)
            print('similar_large_index', similar_large_index)
            print('similar_small_index', similar_small_index)
            print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weights.size()[1] * weights.size()[2] * weights.size()[3]
            '''out * w * h'''
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            '''被选中的filter参数在codebook里置0'''
            print("similar index done")
        else:
            pass
        return codebook
    
    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
    
    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x
    
    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")
        
    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
        print("mask Ready")
    
    def apply(self, weights, amount=0.0, round_to=1):
        indices = []
        if len(self.model_length) == 0:
            self.init_length()
        self.init_mask()
        self.do_similar_mask(weights, amount)
        return indices
    

class RandomStrategy(BaseStrategy):

    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        n_to_prune = int(amount*n) if amount<1.0 else amount
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0: return []
        indices = random.sample( list( range(n) ), k=n_to_prune )
        return indices

class LNStrategy(BaseStrategy):
    def __init__(self, p):
        self.p = p

    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        l1_norm = torch.norm( weights.view(n, -1), p=self.p, dim=1 )
        n_to_prune = int(amount*n) if amount<1.0 else amount 
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0: return []
        threshold = torch.kthvalue(l1_norm, k=n_to_prune).values 
        indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        return indices

class L1Strategy(LNStrategy):
    def __init__(self):
        print('Using L1 Strategy')
        super(L1Strategy, self).__init__(p=1)

class L2Strategy(LNStrategy):
    def __init__(self):
        print('Using L2 Strategy')
        super(L2Strategy, self).__init__(p=2)
