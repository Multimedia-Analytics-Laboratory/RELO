import copy
import os

from lib.config.relo.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.utils import TrackerParams


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir

    yaml_file = os.path.join(prj_dir, "experiments/relo/%s.yaml" % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = copy.deepcopy(cfg)

    params.yaml_name = yaml_name
    params.template_factor = params.cfg.TEST.TEMPLATE_FACTOR
    params.template_size = params.cfg.TEST.TEMPLATE_SIZE
    params.search_factor = params.cfg.TEST.SEARCH_FACTOR
    params.search_size = params.cfg.TEST.SEARCH_SIZE

    params.checkpoint = os.path.join(
        save_dir,
        "checkpoints/train/relo/%s/RELO_ep%04d.pth.tar" % (yaml_name, params.cfg.TEST.EPOCH),
    )
    params.save_all_boxes = False

    return params
