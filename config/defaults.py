from yacs.config import CfgNode as CN

_C = CN()

_C.pid_num = 4
_C.margin = 0.3

_C.cuda = '0'

_C.mode = 'train'
_C.output_path = 'results/'

_C.steps = 100
_C.milestones = [40, 70]
_C.base_learning_rate = 0.008
_C.weight_decay = 1e-4
_C.total_train_epochs = 10
_C.auto_resume_training_from_lastest_steps = False
_C.max_save_model_num = 1
_C.total_train_epochs = 100

_C.market_path = '/home/l/mym/temp/datasets'
_C.duke_path = '/home/wangguanan/datasets/PersonReID/Duke/DukeMTMC-reID/'
_C.msmt_path = '/data/datasets/MSMT17_V1/'
_C.njust_path = '/data/datasets/njust365/'
_C.wildtrack_path = '/data/datasets/Wildtrack_crop_dataset/crop/'
_C.combine_all = False
_C.train_dataset = ['market']
_C.test_dataset = 'market'
_C.image_size = [256, 256]
_C.p= 4
_C.k= 1