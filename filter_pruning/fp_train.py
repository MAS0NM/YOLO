import argparse

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

#### filter pruning related  ####
from fp_utils.utils import *
from search.common import *
import torch_pruning as tp
import fp_test

#### model related ####
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *
from utils.read_write import write_pruning_rate


# warmup = True
# # mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
#     mixed_precision = False  # not installed

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


class Training:

    def __init__(self, hyp, opt, device, mixed_precision=False, savelog=False):

        self.hyp = hyp
        self.opt = opt
        self.device = device
        self.mixed_precision = mixed_precision
        self.savelog = savelog
        self.tb_writer = None
        self.train_path = ''
        self.test_path = ''
        self.nc = 1
        self.name = []
        self.imgsz = 0
        self.imgsz_test = 0
        self.model = None
        self.start_epoch = 0 
        self.best_fitness = 0.0
        self.gs = 0


        ## private
        self.__log_dir = ''
        self.__wdir = ''
        self.__last = ''
        self.__best = ''
        self.__results_file = ''
        self.__pruning_results_file =''

        if self.opt.hyp:  # update hyps
            self.opt.hyp = check_file(self.opt.hyp)  # check file
            with open(self.opt.hyp) as f:
                hyp.update(yaml.load(f, Loader=yaml.FullLoader))  # update hyps
        
        self.opt.img_size.extend([self.opt.img_size[-1]] * (2 - len(self.opt.img_size)))  # extend to 2 sizes (train, test)
        self.opt.total_batch_size = self.opt.batch_size
        self.opt.world_size = 1

        if self.device.type == 'cpu':
            self.mixed_precision = False
        elif self.opt.local_rank != -1:
            # DDP mode
            assert torch.cuda.device_count() > self.opt.local_rank
            torch.cuda.set_device(self.opt.local_rank)
            device = torch.device("cuda", self.opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

            self.opt.world_size = dist.get_world_size()
            assert self.opt.batch_size % self.opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
            self.opt.batch_size = self.opt.total_batch_size // self.opt.world_size

        self.__create_log_dir()

        self.epochs = self.opt.epochs
        self.batch_size = self.opt.batch_size
        self.total_batch_size = self.opt.total_batch_size
        self.weights = self.opt.weights
        self.rank = self.opt.local_rank
        self.nbs = 64  # nominal batch size
        # TODO: Init DDP logging. Only the first process is allowed to log.
        # Since I see lots of print here, the logging configuration is skipped here. We may see repeated outputs.

        init_seeds(2 + self.rank)
        with open(self.opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        self.train_path = data_dict['train']
        self.test_path = data_dict['val']
        self.nc, self.names = (1, ['item']) if self.opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(self.names), self.nc, self.opt.data)  # check

        # Remove previous results
        if self.rank in [-1, 0] and self.savelog:
            for f in glob.glob('*_batch*.jpg') + glob.glob(self.__results_file):
                os.remove(f)

        self.load_model_config()
        self.dataset, self.dataloader, self.testloader = self.__load_dataset()
        self.nb = len(self.dataloader)


    def __del__(self):
        dist.destroy_process_group() if self.rank not in [-1, 0] else None
        torch.cuda.empty_cache()

    def get_model(self):
        return self.model, self.imgsz

    def load_model_config(self):
        self.model = Model(self.opt.cfg, nc=self.nc)

        # Image sizes
        self.gs = int(max(self.model.stride))  # grid size (max stride)
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in self.opt.img_size]  # verify imgsz are gs-multiples

        # # Model parameters
        # # nc: the number of classes
        # self.hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        # self.model.nc = self.nc  # attach number of classes to model
        # self.model.hyp = self.hyp  # attach hyperparameters to model
        # self.model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        # self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(self.device)  # attach class weights
        # self.model.names = self.names


    def load_model_weights(self):
        # Load Model
        with torch_distributed_zero_first(self.rank):
            google_utils.attempt_download(self.weights)

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
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

            # load results
            if ckpt.get('training_results') is not None and self.savelog:
                with open(self.__results_file, 'a') as file:
                    file.write(ckpt['training_results'])  # write results.txt

            # epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.epochs < self.start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                      (self.weights, ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt

    def adaptive_BN(self, in_model=None, deepcp=True):
        if in_model and deepcp:
            model = deepcopy(in_model) 
        elif in_model and not deepcp:
            model  = in_model
        else:
            model = self.model
        #############################
        #######  Adaptive_BN  #######
        #############################
        # print("before adaptive BN........")
        # mapresults, mapss, timess = fp_test.test(self.opt.data,
        #                                      batch_size=self.total_batch_size,
        #                                      imgsz=self.imgsz_test,
        #                                      #  save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
        #                                      model = model,
        #                                      #  model=ema.ema.module if hasattr(self.ema.ema, 'module') else ema.ema,
        #                                      single_cls=self.opt.single_cls,
        #                                      dataloader=self.testloader)
        # # print('%10.4g' * 7 % mapresults + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        # print("\n")
    
        print('Adaptive BN start....\n')
        # losses = AverageMeter()
        model.train()
        mloss_bn = torch.zeros(4, device=self.device)  # mean losses
        with torch.no_grad():
            pbar = enumerate(self.dataloader)
            pbar = tqdm(pbar, total=self.nb)  # progress bar
            # for epoch in range(3):
            for batch, (inputs, targets, paths, _) in pbar:
                # if batch < 50:
                    imgs = inputs.to(self.device, non_blocking=True).float() / 255.0
                    pred = model(imgs)
                    # if self.ema is not None:
                    #     self.ema.update(model)
                    # loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                    # mloss_bn = (mloss_bn * batch + loss_items) / (batch + 1)  # update mean losses
                # else:
                #     break
                # losses.update(loss.item(), imgs.size(0))
                # print('loss.item(): ', loss.item())
    
        # print("after adaptive BN........")
        # mapresults, mapss, timess = fp_test.test(self.opt.data,
        #                                      batch_size=self.total_batch_size,
        #                                      imgsz=self.imgsz_test,
        #                                      #  save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
        #                                      model=model,
        #                                      single_cls=self.opt.single_cls,
        #                                      dataloader=self.testloader)
        # # print('%10.4g' * 7 % mapresults + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        # print("\n")

        return model


    def test(self, in_model=None, deepcp=False):
        if in_model and deepcp:
            model = deepcopy(in_model) 
        elif in_model and not deepcp:
            model  = in_model
        else:
            model = self.model

        results, maps, times = fp_test.test(self.opt.data,
                                         batch_size=self.total_batch_size,
                                         imgsz=self.imgsz_test,
                                        #  save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                         model=model,
                                         single_cls=self.opt.single_cls,
                                         dataloader=self.testloader)
        return results


    def train(self, setting_epochs=None, in_nw=-1, in_model=None, warmup=True, deepcp=False):
        epochs = setting_epochs if setting_epochs else self.epochs
        
        if in_model and deepcp:
            model = deepcopy(in_model) 
        elif in_model and not deepcp:
            model  = in_model
        else:
            model = self.model


        nb = len(self.dataloader)
        assert nb>0, 'Creat Scheduler: dataloader failed to load\n'

        ##### learning rate setting ######
        # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # # lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        # lf = lambda x: (((1 + math.cos(x * math.pi / (epochs*nb))) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # # print('plot lr scheduler start')
        # plot_lr_scheduler(optimizer, scheduler, epochs*nb, log_dir)
        # # print('plot lr scheduler finish')

        if warmup:
            # print('Using Warmup')
            # Scheduler https://arxiv.org/pdf/1812.01187.pdf
            # lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
            self.lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
            if self.savelog:
                plot_lr_scheduler_warmup(self.optimizer, self.scheduler, nb, epochs, self.__log_dir)
        else:
            # Scheduler https://arxiv.org/pdf/1812.01187.pdf
            # lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
            self.lf = lambda x: (((1 + math.cos(x * math.pi / (epochs*nb))) / 2) ** 1.0) * 0.8 + 0.2  # cosine
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
            if self.savelog:
                plot_lr_scheduler(self.optimizer, self.scheduler, epochs*nb, self.__log_dir)


        # Mixed precision training https://github.com/NVIDIA/apex
        if self.mixed_precision:
            # print('Using mixed precison training')
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level='O1', verbosity=0)

        # DP mode: multi-gpu
        if self.device.type != 'cpu' and self.rank == -1 and torch.cuda.device_count() > 1:
            # print('Using DataParallel()')
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        # sync_bn: false
        if self.opt.sync_bn and self.device.type != 'cpu' and self.rank != -1:
            # print('Using SyncBatchNorm()')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)

        # Exponential moving average
        ema = torch_utils.ModelEMA(model) if self.rank in [-1, 0] else None
        # print('Using Exponential moving average')

        # DDP mode:distributed data parallel
        if self.device.type != 'cpu' and self.rank != -1:
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
            # print('Using DDP mode')
        

        # Start training
        # nc: number of classes
        # nb: number of batches(iterations), num_training_pics / batch_size
        # nw: number of warmup iterations, max(3 epochs, 1k iterations)
        t0 = time.time()
        # nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
        nw = in_nw if in_nw>=0 else max(3 * self.nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # print("number of warmup iterations: ", nw)
        # print("epochs: ", nb)
        maps = np.zeros(self.nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        # if self.rank in [0, -1]:
        #     print('Image sizes %g train, %g test' % (self.imgsz, self.imgsz_test))
        #     print('Using %g dataloader workers' % self.dataloader.num_workers)
        #     print('Starting training for %g epochs...' % epochs)

         ####################  epoch  ######################
        # torch.autograd.set_detect_anomaly(True)
        # epoch_exit = False
        best_acc = 0.0
        for epoch in range(self.start_epoch, epochs):  # epoch ------------------------------------------------------------------
            model.train()

            # Update image weights (optional)
            # When in DDP mode, the generated indices will be broadcasted to synchronize dataset.
            # dataset.image_weights: default(false)
            if self.dataset.image_weights:
                print('Update image weights')
                # Generate indices.
                if self.rank in [-1, 0]:
                    w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                    image_weights = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=w)
                    self.dataset.indices = random.choices(range(self.dataset.n), weights=image_weights,
                                                     k=self.dataset.n)  # rand weighted idx
                # Broadcast.
                if self.rank != -1:
                    indices = torch.zeros([self.dataset.n], dtype=torch.int)
                    if self.rank == 0:
                        indices[:] = torch.from_tensor(self.dataset.indices, dtype=torch.int)
                    dist.broadcast(indices, 0)
                    if self.rank != 0:
                        self.dataset.indices = indices.cpu().numpy()

            # Update mosaic border
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(4, device=self.device)  # mean losses
            if self.rank != -1:
                self.dataloader.sampler.set_epoch(self.epoch)
            pbar = enumerate(self.dataloader)
            if self.rank in [-1, 0]:
                print(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'lr'))
                pbar = tqdm(pbar, total=self.nb)  # progress bar
            self.optimizer.zero_grad()

            ######################  batch ##########################
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                # ni: idx of taining pics(global, 0-len(trainset))
                # nb: number of batches(iterations), num_training_pics / batch_size
                # nw: number of warmup iterations, max(3 epochs, 1k iterations)
                ni = i + self.nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

                ############   Warmup    ############
                # Warmup
                if ni <= nw and warmup:
                    # print('---------warmup-----------')
                    xi = [0, nw]  # x interp

                    # y0 = np.interp(x0, xi, yi) 
                    # xi: 横坐标， yi：纵坐标，(xi,yi) 获得一条 y-x 曲线，返回 x0 处的 y0 值
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                    accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.total_batch_size]).round())

                    # print('\t\t\t\t epoch  ni/nw  nbs/total_batch_size ', epoch, ni, nw, nbs, total_batch_size)
                    # print('xi ', xi)    xi  [0, 1000.0]
                    # print('yi ', [1, nbs / total_batch_size])     yi  [1, 1.0]
                    # print('accumulate ', accumulate)    accumulate  1

                    # optimizer.param_groups：  pg0：all else     pg1: weights       pg2: biases
                    # if: consic learning rate   每个image 有一个 x['lr']，x['momentum']
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

                        # print('\t\t\t\tj   lf(epoch)  x[initial_lr]  x[lr]  momentum   |  ', j, lf(epoch), x['initial_lr'], x['lr'], x['momentum'])
                
                # if nw < len(self.dataloader) and ni>=nw:
                #     # epoch_exit = True
                #     break

                ############  Multi-scale input ############
                # multi-scale: default(false)
                if self.opt.multi_scale:
                    # print('Using multi scale')
                    sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)


                #############  Forward  ##################
                # print('forward')
                pred = model(imgs)

                #############  Compute Loss   ################
                loss, loss_items = compute_loss(pred, targets.to(self.device), model)  # scaled by batch_size
                if self.rank != -1:
                    loss *= self.opt.world_size  # gradient averaged between devices in DDP mode
                # 判断数据是否溢出
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                #############   Backward  #################
                if self.mixed_precision:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                ############# Optimize   ##################
                if ni % accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if ema is not None:
                        ema.update(model)

                if not warmup:
                    # Scheduler
                    self.scheduler.step()

                ############  Print    ###################
                if self.rank in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 6 + '   %s') % (
                        '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1], self.optimizer.param_groups[0]['lr'])
                    pbar.set_description(s)

                    ### save train_batch.jpg
                    # Plot 
                    if ni < 3 and self.savelog:
                        f = str(Path(self.log_dir) / ('train_batch%g.jpg' % ni))  # filename
                        result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                        if self.tb_writer and result is not None:
                            self.tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                            tb_writer.add_graph(model, imgs)  # add model to tensorboard

                # end batch ------------------------------------------------------------------------------------------------
            ######################  batch ##########################
            # if epoch_exit:
            #     break

            if warmup:
                # Scheduler
                self.scheduler.step()


            ##################  test mAP  #####################
            # Only the first process in DDP mode is allowed to log or save checkpoints.

            if self.rank in [-1, 0]:
                # mAP
                if ema is not None:
                    ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
                final_epoch = epoch + 1 == epochs

                if not self.opt.notest or final_epoch:  # Calculate mAP
                    savedir = self.__log_dir if self.savelog else ''
                    # print('savedir: ', savedir)
                    # results, maps, times = fp_test.test(opt.data,
                    #                                  batch_size=total_batch_size,
                    #                                  imgsz=imgsz_test,
                    #                                  save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                    #                                  model=model,
                    #                                  single_cls=opt.single_cls,
                    #                                  dataloader=testloader,
                    #                                  save_dir=savedir)

                    results, maps, times = fp_test.test(self.opt.data,
                                                     batch_size=self.total_batch_size,
                                                     imgsz=self.imgsz_test,
                                                     save_json=final_epoch and self.opt.data.endswith(os.sep + 'coco.yaml'),
                                                     model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                     single_cls=self.opt.single_cls,
                                                     dataloader=self.testloader,
                                                     save_dir=savedir)

                    best_acc = max(best_acc, results[2])

                    # Write
                    if self.savelog:
                        with open(self.results_file, 'a') as f:
                            f.write(s + '   %10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)


                    if len(self.opt.name) and self.opt.bucket:
                        os.system('gsutil cp %s gs://%s/results/results%s.txt' % (self.__results_file, self.opt.bucket, self.opt.name))

                    # Tensorboard
                    if self.tb_writer:
                        tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                                'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                        for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                            self.tb_writer.add_scalar(tag, x, epoch)

                    # Update best mAP
                    fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
                    if fi > self.best_fitness:
                        self.best_fitness = fi

                # Save model
                if self.savelog:
                    save = (not self.opt.nosave) or (final_epoch and not self.opt.evolve)
                    if save:
                        with open(self.__results_file, 'r') as f:  # create checkpoint
                            ckpt = {'epoch': epoch,
                                    'best_fitness': self.best_fitness,
                                    'training_results': f.read(),
                                    'model': ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                    'optimizer': None if final_epoch else self.optimizer.state_dict()}

                        # Save last, best and delete
                        torch.save(ckpt, self.last)
                        if (self.best_fitness == fi) and not final_epoch:
                            torch.save(ckpt, self.__best)
                        del ckpt
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training

        if self.rank in [-1, 0]:
            if self.savelog:
                # Strip optimizers
                n = ('_' if len(self.opt.name) and not self.opt.name.isnumeric() else '') + self.opt.name
                fresults, flast, fbest = 'results%s.txt' % n, self.__wdir + 'last%s.pt' % n, self.__wdir + 'best%s.pt' % n
                for f1, f2 in zip([self.__wdir + 'last.pt', self.__wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
                    if os.path.exists(f1):
                        os.rename(f1, f2)  # rename
                        ispt = f2.endswith('.pt')  # is *.pt
                        strip_optimizer(f2) if ispt else None  # strip optimizer
                        os.system('gsutil cp %s gs://%s/weights' % (f2, self.opt.bucket)) if self.opt.bucket and ispt else None  # upload
            # Finish
            if not self.opt.evolve and self.savelog:
                plot_results(save_dir=self.__log_dir)  # save as results.png
            print('%g epochs completed in %.3f hours.\n' % (epoch - self.start_epoch + 1, (time.time() - t0) / 3600))

        return ema.ema.module if hasattr(ema.ema, 'module') else ema.ema, model, best_acc, results


    def create_optimizer(self):
        # Optimizer
        # nbs = 64  # nominal batch size
        # default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
        # all-reduce operation is carried out during loss.backward().
        # Thus, there would be redundant all-reduce communications in a accumulation procedure,
        # which means, the result is still right but the training speed gets slower.
        # TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
        # in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
        accumulate = max(round(self.nbs / self.total_batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= self.total_batch_size * accumulate / self.nbs  # scale weight_decay

        # count number of biases, weights, all else
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else

        if hyp['optimizer'] == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
            self.optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        return self.optimizer


    # def create_scheduler(self, warmup=True, setting_epochs=-1):

    #     epochs = setting_epochs if setting_epochs>=0 else self.epochs
    #     nb = len(self.dataloader)
    #     assert nb>0, 'Creat Scheduler: dataloader failed to load\n'

    #     ##### learning rate setting ######
    #     # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    #     # # lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    #     # lf = lambda x: (((1 + math.cos(x * math.pi / (epochs*nb))) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    #     # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    #     # # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    #     # # print('plot lr scheduler start')
    #     # plot_lr_scheduler(optimizer, scheduler, epochs*nb, log_dir)
    #     # # print('plot lr scheduler finish')

    #     if warmup:
    #         # print('Using Warmup')
    #         # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    #         # lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    #         self.lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    #         self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
    #         # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    #         if self.savelog:
    #             plot_lr_scheduler_warmup(self.optimizer, self.scheduler, nb, epochs, self.__log_dir)
    #     else:
    #         # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    #         # lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    #         self.lf = lambda x: (((1 + math.cos(x * math.pi / (epochs*nb))) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    #         self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
    #         # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    #         if self.savelog:
    #             plot_lr_scheduler(self.optimizer, self.scheduler, epochs*nb, self.__log_dir)


    def __create_log_dir(self):
        if self.opt.local_rank in [-1, 0] and self.savelog:
            print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            self.tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', self.opt.name))
        else:
            self.tb_writer = None

        if self.savelog:
            self.__log_dir = self.tb_writer.log_dir if self.tb_writer else 'runs/evolution'  # run directory
            self.__wdir = str(Path(self.__log_dir) / 'weights') + os.sep  # weights directory
            os.makedirs(self.__wdir, exist_ok=True)
            self.__last = self.__wdir + 'last.pt'
            self.__best = self.__wdir + 'best.pt'
            self.__results_file = self.__log_dir + os.sep + 'results.txt'
            self.__pruning_results_file = self.__log_dir + os.sep + 'pruning_results.txt'

            # Save run settings
            with open(Path(self.__log_dir) / 'hyp.yaml', 'w') as f:
                yaml.dump(self.hyp, f, sort_keys=False)
            with open(Path(self.__log_dir) / 'opt.yaml', 'w') as f:
                yaml.dump(vars(self.opt), f, sort_keys=False)
            
            if self.savelog:
                with open(self.__results_file, 'w') as file:
                    # write results.txt
                    file.write('#\t\tEpoch   gpu_mem  GIoU_loss  obj _loss  cls_loss    total       \
                                targets   img_size      lr                P           R           \
                                mAP@.5      mAP@.5:.95    val/giou_loss  val/obj_loss  val/cls_loss\n')  


    def __load_dataset(self):
        # Trainloader
        dataloader, dataset = create_dataloader(self.train_path, self.imgsz, self.batch_size, self.gs, self.opt, 
                                                hyp=self.hyp, augment=True,
                                                cache=self.opt.cache_images, rect=self.opt.rect, local_rank=self.rank,
                                                world_size=self.opt.world_size)
        # np.concatenate 一次完成多个数组的拼接
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches, num_training_pics / batch_size
        assert mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' \
                                                    % (mlc, self.nc, self.opt.data, self.nc - 1)

        # Testloader
        if self.rank in [-1, 0]:
            # local_rank is set to -1. Because only the first process is expected to do evaluation.
            testloader = create_dataloader(self.test_path, self.imgsz_test, self.total_batch_size, self.gs, self.opt, 
                                            # hyp=None, augment=False, cache=False, pad=0.5, rect=True)
                                           hyp=self.hyp, augment=False,
                                           cache=self.opt.cache_images, rect=True, local_rank=-1, world_size=self.opt.world_size)[0]

            


        # Model parameters
        # nc: the number of classes
        self.hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        self.model.class_weights = labels_to_class_weights(dataset.labels, self.nc).to(self.device)  # attach class weights
        self.model.names = self.names

        # Class frequency
        if self.rank in [-1, 0]:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.
            # model._initialize_biases(cf.to(device))
            if self.savelog:
                plot_labels(labels, save_dir=self.__log_dir)
            if self.tb_writer:
                # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                self.tb_writer.add_histogram('classes', c, 0)

            # Check anchors
            if not self.opt.noautoanchor:
                check_anchors(dataset, model=self.model, thr=self.hyp['anchor_t'], imgsz=self.imgsz)
        
        return dataset, dataloader, testloader

    # def __train_setting(self, in_model=None, deepcp=False):
    #     if in_model and deepcp:
    #         model = deepcopy(in_model) 
    #     elif in_model and not deepcp:
    #         model  = in_model
    #     else:
    #         model = self.model

    #     # Mixed precision training https://github.com/NVIDIA/apex
    #     if self.mixed_precision:
    #         # print('Using mixed precison training')
    #         model, self.optimizer = amp.initialize(model, self.optimizer, opt_level='O1', verbosity=0)

    #     # DP mode: multi-gpu
    #     if self.device.type != 'cpu' and self.rank == -1 and torch.cuda.device_count() > 1:
    #         # print('Using DataParallel()')
    #         model = torch.nn.DataParallel(model)

    #     # SyncBatchNorm
    #     # sync_bn: false
    #     if self.opt.sync_bn and self.device.type != 'cpu' and self.rank != -1:
    #         # print('Using SyncBatchNorm()')
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)

    #     # Exponential moving average
    #     ema = torch_utils.ModelEMA(model) if self.rank in [-1, 0] else None
    #     # print('Using Exponential moving average')

    #     # DDP mode:distributed data parallel
    #     if self.device.type != 'cpu' and self.rank != -1:
    #         model = DDP(model, device_ids=[self.rank], output_device=self.rank)
    #         # print('Using DDP mode')

    #     return model, ema.ema.module if hasattr(ema.ema, 'module') else ema.ema

