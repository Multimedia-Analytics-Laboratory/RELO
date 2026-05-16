import copy

from easydict import EasyDict as edict
import yaml

"""Default configuration for RELO policy/value training."""

cfg = edict()

# MODEL
cfg.MODEL = edict()

cfg.MODEL.TTOKEN = edict()
cfg.MODEL.TTOKEN.TYPE = "regist"
cfg.MODEL.TTOKEN.PROJ = True
cfg.MODEL.TTOKEN.BLOCK_IDX = [0]
cfg.MODEL.TTOKEN.MEM_TRANS_LEN = 2
cfg.MODEL.TTOKEN.NUM = 49
cfg.MODEL.TTOKEN.DETACH = True

cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "fastitpnb"
cfg.MODEL.ENCODER.DROP_PATH = 0
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "pretrained/itpn/fast_itpn_base_clipl_e1600.pt"
cfg.MODEL.ENCODER.USE_CHECKPOINT = False
cfg.MODEL.ENCODER.STRIDE = 16
cfg.MODEL.ENCODER.POS_TYPE = "index"
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = True

cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "CENTER"
cfg.MODEL.DECODER.NUM_CHANNELS = 256

cfg.MODEL.VALUE = edict()
cfg.MODEL.VALUE.TYPE = "CENTER"

cfg.MODEL.POLICY = edict()
cfg.MODEL.POLICY.LOAD_DECODER = False

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1
cfg.TRAIN.FREEZE_ENCODER = False
cfg.TRAIN.ENCODER_OPEN = []
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.PRV_CKPT = None
cfg.TRAIN.RL_WEIGHT = 1.0
cfg.TRAIN.AUC_REWARD_WEIGHT = 1.0
cfg.TRAIN.IOU_REWARD_WEIGHT = 1.0
cfg.TRAIN.ADV_NORM = True
cfg.TRAIN.TYPE = "normal"
cfg.TRAIN.PRETRAINED_PATH = None

cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
cfg.DATA.REAL_REGION = False
cfg.DATA.JOINT_AUG = False

cfg.DATA.USE_NLP = edict()
cfg.DATA.USE_NLP.LASOT = False
cfg.DATA.USE_NLP.GOT10K = False
cfg.DATA.USE_NLP.COCO = False
cfg.DATA.USE_NLP.TRACKINGNET = False
cfg.DATA.USE_NLP.VASTTRACK = False
cfg.DATA.USE_NLP.REFCOCOG = False
cfg.DATA.USE_NLP.TNL2K = False
cfg.DATA.USE_NLP.OTB99 = False
cfg.DATA.USE_NLP.DEPTHTRACK = False
cfg.DATA.USE_NLP.LASHER = False
cfg.DATA.USE_NLP.VISEVENT = False

cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000

cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5

cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 500
cfg.TEST.WINDOW = False
cfg.TEST.NUM_TEMPLATES = 1

cfg.TEST.WINDOW_FACTOR = edict()
cfg.TEST.WINDOW_FACTOR.DEFAULT = 2.0

cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.DEFAULT = 999999

cfg.TEST.UPDATE_THRESHOLD = edict()
cfg.TEST.UPDATE_THRESHOLD.DEFAULT = 999999

_DEFAULT_CFG = copy.deepcopy(cfg)
_DYNAMIC_TEST_MAPS = {
    ("TEST", "WINDOW_FACTOR"),
    ("TEST", "UPDATE_INTERVALS"),
    ("TEST", "UPDATE_THRESHOLD"),
}


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg, path=()):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v, path + (k,))
            elif path in _DYNAMIC_TEST_MAPS:
                base_cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    cfg.clear()
    cfg.update(copy.deepcopy(_DEFAULT_CFG))
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
