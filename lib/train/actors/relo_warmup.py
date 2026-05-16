from .base_actor import BaseActor


class RELOWarmupActor(BaseActor):
    """Actor for regression warmup training."""

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_list = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_list, data)

        return loss, status

    def forward_pass(self, data):
        mem_trans_len = self.cfg.MODEL.TTOKEN.MEM_TRANS_LEN
        n_search, b = data['search_images'].shape[:2]
        assert n_search % mem_trans_len == 0
        n_group = n_search // mem_trans_len
        batch_group = n_group * b

        search_seq = data['search_images'].view(n_group, mem_trans_len, b, *data['search_images'].shape[2:])
        search_seq = search_seq.permute(1, 0, 2, 3, 4, 5).contiguous().view(mem_trans_len, batch_group,
                                                                            *data['search_images'].shape[2:])

        template_list = [t for t in data['template_images']]
        template_anno_list = [a for a in data['template_anno']]
        template_list = [t.repeat(n_group, 1, 1, 1) for t in template_list]
        template_anno_list = [a.repeat(n_group, 1) for a in template_anno_list]

        out_list = [None] * n_search
        ttokens = None

        for t in range(mem_trans_len):
            search = search_seq[t]

            enc_opt, ttokens = self.net(
                template_list=template_list,
                template_anno_list=template_anno_list,
                search_list=[search],
                ttokens=ttokens,
                mode='encoder'
            )

            out_dict = self.net(feature=enc_opt, mode='decoder')

            if self.cfg.MODEL.TTOKEN.DETACH and ttokens is not None:
                ttokens = [tok.detach() for tok in ttokens]

            splitted = [{k: v[i * b:(i + 1) * b] for k, v in out_dict.items()} for i in range(n_group)]

            for i in range(n_group):
                index = i * mem_trans_len + t
                out_list[index] = splitted[i]

        return out_list

    def compute_losses(self, pred_list, gt_dict, return_status=True):
        import torch
        from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
        from lib.utils.heapmap_utils import generate_heatmap

        total_status = {}
        device = pred_list[0]['pred_boxes'].device
        base_loss = torch.tensor(0., dtype=torch.float, device=device)

        gt_gaussian_maps = generate_heatmap(
            gt_dict['search_anno'],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.ENCODER.STRIDE)

        for i in range(len(pred_list)):
            gt_bbox = gt_dict['search_anno'][i]
            gt_gaussian_map = gt_gaussian_maps[i].unsqueeze(1)

            pred_boxes = pred_list[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")

            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat(1, num_queries, 1).view(-1, 4).clamp(0.0, 1.0)

            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss = pred_boxes_vec.new_tensor(0.0)
                iou = pred_boxes_vec.new_zeros(pred_boxes_vec.shape[0])

            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

            if 'score_map' in pred_list[i]:
                location_loss = self.objective['focal'](pred_list[i]['score_map'], gt_gaussian_map)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)

            loss = (self.loss_weight['giou'] * giou_loss +
                    self.loss_weight['l1'] * l1_loss +
                    self.loss_weight['focal'] * location_loss)

            base_loss += loss

            if return_status:
                mean_iou = iou.detach().mean()
                status = {
                    f"Frame{i}/Loss": loss.item(),
                    f"Frame{i}/GIoU": giou_loss.item(),
                    f"Frame{i}/L1": l1_loss.item(),
                    f"Frame{i}/Focal": location_loss.item(),
                    f"Frame{i}/IoU": mean_iou.item()
                }
                total_status.update(status)

        total_loss = base_loss / len(pred_list)

        if return_status:
            total_status["Loss/Total"] = total_loss.item()
            return total_loss, total_status
        else:
            return total_loss
