from .base_actor import BaseActor


class RELOActor(BaseActor):
    """Actor for RELO policy/value training."""

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg
        self.auc_r_weight = cfg.TRAIN.AUC_REWARD_WEIGHT
        self.iou_r_weight = cfg.TRAIN.IOU_REWARD_WEIGHT
        self.adv_norm = cfg.TRAIN.ADV_NORM

    def _unwrap(self, net):
        import torch

        return net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net

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

        out_list = self.forward_pass(data)

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

        total_status = {}

        logp_list = []
        reward_list = []
        iou_sampling_list = []
        value_list = []

        for i in range(len(pred_list)):
            gt_bbox = gt_dict['search_anno'][i]
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(0.0, 1.0)
            policy_logits = pred_list[i]['policy_logits']
            for output_name in ("policy_logits", "size_map", "offset_map", "value"):
                if torch.isnan(pred_list[i][output_name]).any():
                    raise ValueError("Network outputs is NAN! Stop Training")
            dist = torch.distributions.Categorical(logits=policy_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            core_net = self.unwrap(self.net)
            bbox_sampling = core_net.decoder.cal_bbox_sampling(idx=action,
                                                               size_map=pred_list[i]['size_map'],
                                                               offset_map=pred_list[i]['offset_map'])
            bbox_sampling_vec = box_cxcywh_to_xyxy(bbox_sampling).view(-1, 4)
            _, iou_reward = self.objective['giou'](bbox_sampling_vec, gt_boxes_vec)

            iou_sampling_list.append(iou_reward.detach())
            reward_list.append(self.iou_r_weight * iou_reward.detach())
            logp_list.append(log_prob)
            value_list.append(pred_list[i]['value'])

            if return_status:
                mean_iou_reward = iou_reward.detach().mean()
                status = {
                    f"Frame{i}/IoU_reward": mean_iou_reward.item()
                }
                total_status.update(status)

        auc_reward = self.compute_auc(iou_sampling_list)
        reward_list = [r + auc_reward * self.auc_r_weight for r in reward_list]
        rewards = torch.stack(reward_list)
        values = torch.stack(value_list)
        logps = torch.stack(logp_list)
        advantages = rewards - values
        returns = rewards
        if self.adv_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(logps - logps.detach())
        policy_loss = -(ratio * advantages.detach()).mean()
        value_loss = ((values - returns.detach()) ** 2).mean()
        rl_loss = policy_loss + 0.5 * value_loss
        total_loss = rl_loss * self.loss_weight['rl']

        if return_status:
            mean_auc_reward = auc_reward.detach().mean()
            total_status["AUC_reward"] = mean_auc_reward.item()
            total_status["Loss/Policy"] = policy_loss.item()
            total_status["Loss/Value"] = value_loss.item()
            total_status["Loss/RL"] = rl_loss.item()
            total_status["Loss/Total"] = total_loss.item()
            return total_loss, total_status
        else:
            return total_loss

    def unwrap(self, net):
        return self._unwrap(net)

    def compute_auc(self, seq_ious, num_thresholds=21):
        import torch

        iou_mat = torch.stack(seq_ious, dim=1)
        thresholds = torch.linspace(0, 1, steps=num_thresholds, device=iou_mat.device)

        iou_expanded = iou_mat[:, None, :]
        threshold_expanded = thresholds[None, :, None]

        success_mat = (iou_expanded > threshold_expanded).float()

        success_rate = success_mat.mean(dim=2)
        auc_per_sample = torch.trapz(success_rate, thresholds, dim=1)
        return auc_per_sample.detach()

    def fix_bn(self):
        import torch

        for name, module in self.net.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                if ("policy_model" not in name) and ("value_model" not in name):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
