import unittest
import sys
import os

####### Pruning ########
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_pruning as tp
from torch_pruning.pruning import filter_pruning
from fp_utils.utils import *
from fp_train import Training
import fp_test as test

####### YOLO ########
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *
from utils.read_write import write_pruning_rate

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

class opt:
    cfg = './models/yolov5s_ori.yaml' # model config file
    data = 'data/quexian.yaml'  # dataset dir
    hyp = ''
    epochs = 10
    batch_size = 64
    img_size = [320, 320]
    rect = False
    resume = False
    nosave = False
    notest = False
    noautoanchor = False
    evolve = False
    bucket = ''
    cache_images = False
    weights = 'weights/DET-biaopan_quexian-20211028.pt'
    name = ''
    device = "0"
    multi_scale = False
    single_cls = False
    sync_bn = False
    local_rank = -1
    total_batch_size = 64
    world_size = 1


a0=[0.2332285241228435, 0.005240557151848889, 0.044444941991939234, 0.21329944746540028, 0.2541464527585162, 0.04752974723220944, 0.025273343258043784, 
        0.21000552805534922, 0.05288166868659055, 0.005539077131060246, 0.2738718265187181, 0.06034298552859498, 0.09885564464638172, 0.02278207277685376, 
        0.1717659428649081, 0.33919663948676915, 0.03611436241256918, 0.13774653341658505, 0.04750180575146392, 0.006114635117233128, 0.02940853124533043, 
        0.09430261614457797, 0.07534657636801811, 0.31544903767242016, 0.055915273281901975, 0.07576031377922823, 0.06103877360926921, 0.05181785560453499, 
        0.08430707886557333, 0.1851928671274287, 0.25105776450218054, 0.2505821822597518, 0.044531644700999615, 0.08635215442255659, 0.06348738646990794, 
        0.10546245565522107, 0.22664113541194525, 0.11571253712616958, 0.01699588727617871, 0.13071600938565878, 0.18632473642769257, 0.05445987664423592, 
        0.20988204026308177, 0.03044817939325877, 0.03830396732982107, 0.11360517742927498, 0.04900417964150457, 0.047126518637486674, 0.036378474196155176, 
        0.06900484310006112, 0.10247085653863863, 0.029770059340275812, 0.04606831872708361, 0.1930410277214638, 0.027601935494318587, 0.023883896757579706, 
        0.0174696613505956, 0.014514978780645827, 0.0850748876107997, 0.15930633138325243, 0.0069422270317072805, 0.02798665202552968, 0.022421087349795445, 
        0.057779754537764155, 0.06839831076264306, 0.23784963920624466, 0.40893242989518896, 0.1778806125405771]
a1=[0.0016378075679704, 0.015030415543571388, 0.19773796557320278, 0.2997781589683227, 0.09038114373771584, 0.15936008677488422, 0.14996070199480024, 
        0.002220253167448488, 0.0007829722977594147, 0.17142212778320987, 0.022847249333919184, 0.14149952157611875, 0.08555540497069046, 0.07377902353825089, 
        0.025994233345097286, 0.002194699314684907, 0.07172458293832328, 0.13190421523117346, 0.08001816063782587, 0.03886969088664369, 0.10824460102882127, 
        0.15358205341719983, 0.02099904235397302, 0.04588758769853538, 0.1336518028010362, 0.15346291316868263, 0.017964180912819906, 0.01724158420827028, 
        0.019279619502178397, 0.42707119745306643, 0.05750798946951994, 0.04907416027389529, 0.06378451540183955, 0.008849515102630833, 0.1659509388695413, 
        0.1103870514507725, 0.0392928000451928, 0.04887665771036867, 0.18496917776199767, 0.06944074747275532, 0.1994484207950644, 0.003224600926379068, 
        0.028266542729200424, 0.14334902846315933, 0.07810567727185652, 0.003544266275301408, 0.1520166949377355, 0.1314459258112866, 0.3638759528436391, 
        0.1967525341579988, 0.34644544416350914, 0.13958666204266476, 0.11002909654463822, 0.052755686766604144, 0.016757010745821765, 0.02380439560923468, 
        0.003739329280717312, 0.06543016175562112, 0.1806016444229635, 0.042181394882202655, 0.12651271593378097, 0.02057659453903148, 0.13288299440901324, 
        0.06619069151305648, 0.057256181377797095, 0.2349417665561081, 0.05111617698345167, 0.06518245428267223]

pruning_rates = []
pruning_rates.append(a0)
pruning_rates.append(a1)

# Model FLOPs = 2165.31M
# Model PARAMs = 7.25M
init_model_params = 7246000
init_model_flops = 2165311000
init_acc = 0.381

a1_pruned_model_params = 5519679
a1_pruned_model_flops = 1658388000
a1_pruned_acc = 0.00751
a1_adaptive_bn_acc = 0.22923360370239854
a1_retrain_50_iteration = 0.23834202568452878

a2_pruned_model_params = 5906617
a2_pruned_model_flops = 1732355200
a2_pruned_acc = 0.21
a2_adaptive_bn_acc = 0.20147917349365646
a2_retrain_50_iteration = 0.204467095821735

pruned_model_params=[a1_pruned_model_params, a2_pruned_model_params]
pruned_model_flops=[a1_pruned_model_flops, a2_pruned_model_flops]
pruned_acc = [a1_pruned_acc, a2_pruned_acc]
adaptive_bn_acc = [a1_adaptive_bn_acc, a2_adaptive_bn_acc]
retrain_50_iteration = [a1_retrain_50_iteration, a2_retrain_50_iteration]

class TestTrain(unittest.TestCase):


    def test_pruning(self):
        mixed_precision = False
        device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)

        trainer = Training(hyp, opt, device, mixed_precision=mixed_precision, savelog=False) 
        trainer.create_optimizer()
        trainer.load_model_weights()
        trainer.create_scheduler(warmup=True)
    
        model_init, imgsz = trainer.get_model()

        ######## init model ##########
        layer_params = calc_model_parameters(model_init)
        layer_flops = calc_model_flops(model_init, input_size=imgsz, mul_add=False)
        total_params = sum(layer_params)
        total_flops = sum(layer_flops)

        self.assertEqual(int(total_flops), init_model_flops)
        self.assertEqual(int(total_params), init_model_params)

        model_init.to(device)
        results = trainer.test(in_model=model_init)
        print(results[2])
        self.assertTrue(abs(results[2]-init_acc) < 1e-2)

        for idx, fp_rate in enumerate(pruning_rates):
            
            ######## after pruning ##########
            pruned_model = filter_pruning(model_init, imgsz, fp_rate)
            pruned_model = pruned_model.to(device)

            layer_params = calc_model_parameters(pruned_model)
            layer_flops = calc_model_flops(pruned_model, input_size=imgsz, mul_add=False)
            total_params = sum(layer_params)
            total_flops = sum(layer_flops)

            self.assertEqual(int(total_flops), pruned_model_flops[idx])
            self.assertEqual(int(total_params), pruned_model_params[idx])

            results = trainer.test(in_model=pruned_model)
            print(results[2])
            self.assertTrue(abs(results[2]-pruned_acc[idx]) < 1e-2)


            ######## after adaptive BN ##########
            ada_bn_model = trainer.adaptive_BN(in_model=pruned_model, deepcp=True)

            layer_params = calc_model_parameters(ada_bn_model)
            layer_flops = calc_model_flops(ada_bn_model, input_size=imgsz, mul_add=False)
            total_params = sum(layer_params)
            total_flops = sum(layer_flops)

            self.assertEqual(int(total_flops), pruned_model_flops[idx])
            self.assertEqual(int(total_params), pruned_model_params[idx])

            results = trainer.test(in_model=ada_bn_model)
            print(results[2])
            self.assertTrue(abs(results[2]-adaptive_bn_acc[idx]) < 5e-2)



            ######## after retrain 50 iteration ##########
            retrain_model, best_acc = trainer.train(setting_epochs=1, in_nw=50, in_model=pruned_model, warmup=True)

            layer_params = calc_model_parameters(retrain_model)
            layer_flops = calc_model_flops(retrain_model, input_size=imgsz, mul_add=False)
            total_params = sum(layer_params)
            total_flops = sum(layer_flops)

            self.assertEqual(int(total_flops), pruned_model_flops[idx])
            self.assertEqual(int(total_params), pruned_model_params[idx])

            results = trainer.test(in_model=retrain_model)
            print(results[2])
            self.assertTrue(abs(results[2]-retrain_50_iteration[idx]) < 5e-2)


if __name__ == '__main__':
    unittest.main()