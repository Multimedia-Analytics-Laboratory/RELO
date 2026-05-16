import re
from collections.abc import Mapping

import cv2
import torch

from lib.models.relo import build_relo
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.utils import Preprocessor, sample_target, transform_image_to_crop
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box


_MISSING = object()
_DATASET_ALIASES = {
    "LASOTEXTENSIONSUBSET": "LASOT_EXTENSION_SUBSET",
    "LASOT_EXT": "LASOT_EXTENSION_SUBSET",
    "LASOTEXT": "LASOT_EXTENSION_SUBSET",
    "GOT10KTEST": "GOT10K_TEST",
    "GOT10K": "GOT10K_TEST",
}


def normalize_dataset_name(dataset_name):
    if dataset_name is None:
        return "DEFAULT"
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(dataset_name).upper()).strip("_")
    if not key:
        return "DEFAULT"
    return _DATASET_ALIASES.get(key, _DATASET_ALIASES.get(key.replace("_", ""), key))


def _mapping_items(value):
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return list(value.items())
    return None


def _get_mapping_value(value, key):
    items = _mapping_items(value)
    if items is None:
        return _MISSING
    normalized_key = normalize_dataset_name(key)
    for item_key, item_value in items:
        if normalize_dataset_name(item_key) == normalized_key:
            return item_value
    return _MISSING


def resolve_dataset_value(value, dataset_name, default_key="DEFAULT"):
    items = _mapping_items(value)
    if items is None:
        return value

    dataset_value = _get_mapping_value(value, dataset_name)
    if dataset_value is not _MISSING:
        return dataset_value

    default_value = _get_mapping_value(value, default_key)
    if default_value is not _MISSING:
        return default_value

    raise KeyError("No dataset override or DEFAULT value found for %s" % dataset_name)


def _get_nested_value(config, path):
    value = config
    for part in path:
        if isinstance(value, Mapping):
            value = value[part]
        elif hasattr(value, "__getitem__") and part in value:
            value = value[part]
        else:
            value = getattr(value, part)
    return value


def resolve_nested_dataset_value(config, path, dataset_name, default_key="DEFAULT"):
    if isinstance(path, str):
        path = tuple(path.split("."))
    return resolve_dataset_value(_get_nested_value(config, path), dataset_name, default_key)


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_state_dict(path):
    checkpoint = _load_checkpoint(path)
    if not isinstance(checkpoint, Mapping) or "net" not in checkpoint:
        raise ValueError("Malformed checkpoint %s: expected key 'net'." % path)
    return checkpoint["net"]


def _score_as_float(score):
    if torch.is_tensor(score):
        return float(score.detach().reshape(-1)[0].item())
    return float(score)


class RELO(BaseTracker):
    def __init__(self, params, dataset_name):
        super(RELO, self).__init__(params)
        network = build_relo(params.cfg)
        network.load_state_dict(load_checkpoint_state_dict(self.params.checkpoint), strict=True)

        self.cfg = params.cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self.device)
        self.network.eval()
        self.preprocessor = Preprocessor(device=self.device)
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        self.window_factor = resolve_nested_dataset_value(
            self.cfg, ("TEST", "WINDOW_FACTOR"), dataset_name
        )
        if self.cfg.TEST.WINDOW:
            self.output_window = hann2d(
                torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True
            ).to(self.device)

        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.debug = getattr(params, "debug", 0)
        self.frame_id = 0

        self.update_intervals = resolve_nested_dataset_value(
            self.cfg, ("TEST", "UPDATE_INTERVALS"), dataset_name
        )

        self.update_threshold = resolve_nested_dataset_value(
            self.cfg, ("TEST", "UPDATE_THRESHOLD"), dataset_name
        )

        self.ttokens = None

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor = sample_target(
            image,
            info["init_bbox"],
            self.params.template_factor,
            output_sz=self.params.template_size,
        )
        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        self.state = info["init_bbox"]
        prev_box_crop = transform_image_to_crop(
            torch.tensor(info["init_bbox"], device=self.device, dtype=torch.float32),
            torch.tensor(info["init_bbox"], device=self.device, dtype=torch.float32),
            resize_factor,
            torch.tensor([self.params.template_size, self.params.template_size], device=self.device),
            normalize=True,
        )
        template_anno = prev_box_crop.to(template.device).unsqueeze(0)
        self.template_anno_list = [template_anno] * self.num_template
        self.frame_id = 0
        self.seq_name = info.get("seq_name", "")

    def track(self, image, info: dict = None):
        info = {} if info is None else info
        h, w, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor = sample_target(
            image,
            self.state,
            self.params.search_factor,
            output_sz=self.params.search_size,
        )
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            enc_opt, self.ttokens = self.network.forward_encoder(
                self.template_list, [search], self.template_anno_list, self.ttokens
            )
            out_dict = self.network.inference_decoder(feature=enc_opt)

        pred_score_map = self._select_location_map(out_dict)

        if self.cfg.TEST.WINDOW:
            response = (self.output_window ** self.window_factor) * pred_score_map
        else:
            response = pred_score_map

        if "size_map" in out_dict:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict["size_map"], out_dict["offset_map"], return_score=True
            )
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict["offset_map"], return_score=True
            )

        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), h, w, margin=10)

        self._maybe_update_template(image, conf_score)
        self._show_debug_frame(image, conf_score)

        return {"target_bbox": self.state, "best_score": conf_score}

    def _select_location_map(self, out_dict):
        return out_dict["policy_logits"]

    def _maybe_update_template(self, image, conf_score):
        if self.num_template <= 1:
            return
        if self.frame_id % int(self.update_intervals) != 0:
            return
        if _score_as_float(conf_score) <= float(self.update_threshold):
            return

        z_patch_arr, resize_factor = sample_target(
            image,
            self.state,
            self.params.template_factor,
            output_sz=self.params.template_size,
        )
        template = self.preprocessor.process(z_patch_arr)
        self.template_list.append(template)
        if len(self.template_list) > self.num_template:
            self.template_list.pop(1)

        prev_box_crop = transform_image_to_crop(
            torch.tensor(self.state, device=self.device, dtype=torch.float32),
            torch.tensor(self.state, device=self.device, dtype=torch.float32),
            resize_factor,
            torch.tensor([self.params.template_size, self.params.template_size], device=self.device),
            normalize=True,
        )
        self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
        if len(self.template_anno_list) > self.num_template:
            self.template_anno_list.pop(1)

    def _show_debug_frame(self, image, conf_score):
        if self.debug != 1:
            return
        x1, y1, box_w, box_h = self.state
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(
            image_bgr,
            (int(x1), int(y1)),
            (int(x1 + box_w), int(y1 + box_h)),
            color=(0, 0, 255),
            thickness=2,
        )
        text = "conf: %.3f" % _score_as_float(conf_score)
        cv2.putText(
            image_bgr,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3,
        )
        cv2.imshow("RELO", cv2.resize(image_bgr, (500, 500)))
        cv2.waitKey(1)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, box_w, box_h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * box_w, cy_real - 0.5 * box_h, box_w, box_h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, box_w, box_h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack(
            [cx_real - 0.5 * box_w, cy_real - 0.5 * box_h, box_w, box_h], dim=-1
        )


def get_tracker_class():
    return RELO
