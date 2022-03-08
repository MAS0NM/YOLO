import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import check_img_size
from utils.read_write import write_pruning_rate



class FP_Model_YOLO:

    def __init__(self, device, opt, nc): 

        self.config = opt.cfg
        self.device = device
        self.weights = self.weights = self.opt.weights

        self.model = Model(opt.cfg, nc=nc)

        # Image sizes
        self.gridsz = int(max(self.model.stride))  # grid size (max stride)
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        self.imgsz, self.test_imgsz = [check_img_size(x, self.gridsz) for x in opt.img_size]  # verify imgsz are gs-multiples

    def get_model(self):
        return self.model

    def get_imgsz(self):
        return self.imgsz, self.test_imgsz

    def get_gridsz(self):
        return self.gridsz
        
    def load_model_weights(self, opt, optimizer, savelog=False, results_file=''):
        # Load Model
        with torch_distributed_zero_first(opt.rank):
            google_utils.attempt_download(opt.weights)

        start_epoch, best_fitness = 0, 0.0
        if self.weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint

            # load model
            try:
                exclude = ['anchor']  # exclude keys
                ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                                 if k in self.model.state_dict() and not any(x in k for x in exclude)
                                 and self.model.state_dict()[k].shape == v.shape}
                self.model.load_state_dict(ckpt['model'], strict=False)
                print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(self.model.state_dict()), self.weights))
            except KeyError as e:
                s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                    "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                    % (self.weights, opt.cfg, self.weights, self.weights)
                raise KeyError(s) from e

            # load optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # load results
            if ckpt.get('training_results') is not None and savelog:
                with open(results_file, 'a') as file:
                    file.write(ckpt['training_results'])  # write results.txt

            # epochs
            start_epoch = ckpt['epoch'] + 1
            if opt.epochs < start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                      (self.weights, ckpt['epoch'], opt.epochs))
                opt.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt

        return start_epoch, best_fitness
