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
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}
        
    def get_filter_codebook(self, weight_torch, compress_rate, length):
        '''
            weight_torch: model.parameters()[index].data
            length: model_length[index]
            weight_torch.size()
        '''
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            print("filter codebook done")
        else:
            pass
        return codebook
    
    
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            print('filter_large_index', filter_large_index)
            print('filter_small_index', filter_small_index)
            print('similar_sum', similar_sum)
            print('similar_large_index', similar_large_index)
            print('similar_small_index', similar_small_index)
            print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
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
    
    def apply(self, weights, amount=0.0, round_to=1)-> Sequence[int]:  #return index
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
