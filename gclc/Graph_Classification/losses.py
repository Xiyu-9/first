from __future__ import print_function

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # 1) 检查 contrast_feature 是否含 NaN
        if torch.isnan(contrast_feature).any():
            print("contrast_feature has NaN")

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # compute logits
        # 对特征归一化
        anchor_feature = F.normalize(anchor_feature, p=2, dim=1)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        # 2) compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        if torch.isnan(anchor_dot_contrast).any():
            print("anchor_dot_contrast has NaN")

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        if torch.isnan(logits).any():
            print("logits has NaN after subtracting max")

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # 3) 检查 mask
        if torch.isnan(mask).any():
            print("mask has NaN")

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        if torch.isnan(logits_mask).any():
            print("logits_mask has NaN")

        mask = mask * logits_mask
        if torch.isnan(mask).any():
            print("mask has NaN after mask-out self-contrast")

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        if torch.isnan(exp_logits).any():
            print("exp_logits has NaN (check for large logits?)")

        sum_exp_logits = exp_logits.sum(1, keepdim=True)
        # 防止 log(0) 计算导致 NaN
        if (sum_exp_logits == 0).any():
            print("Warning: sum_exp_logits contains zero values!")
        sum_exp_logits = sum_exp_logits + 1e-10 

        log_prob = logits - torch.log(sum_exp_logits)
        if torch.isnan(log_prob).any():
            print("log_prob has NaN")

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        if torch.isnan(mask_pos_pairs).any():
            print("mask_pos_pairs has NaN before torch.where")
        # 防止除法 0/0 计算导致 NaN
        if (mask_pos_pairs == 0).any():
            print("Warning: mask_pos_pairs contains zero values!")
            
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        if torch.isnan(mask_pos_pairs).any():
            print("mask_pos_pairs has NaN after torch.where")

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        if torch.isnan(mean_log_prob_pos).any():
            print("Error: mean_log_prob_pos has NaN after fix!")

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # 在 view 之前检查
        #if torch.isnan(loss).any():
        #    print("loss has NaN before view")
        #    print("loss values:", loss)

        #print("Before view: loss min: {:.6f}, max: {:.6f}, mean: {:.6f}".format(
        #    loss.min().item(), loss.max().item(), loss.mean().item()))
        loss = loss.view(anchor_count, batch_size).mean()
        #loss = loss.contiguous().view(anchor_count, batch_size).mean()
        #print("After view and mean: loss: {:.6f}".format(loss.item()))
        #if torch.isnan(loss):
        #    print("loss has NaN after view and mean")

        return loss

  
class ImprovedClusterLoss(nn.Module):
    def __init__(self, entropy_weight=0.5, repulsion_weight=1.0, temperature=0.1):
        super(ImprovedClusterLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss().cuda()
        self.entropy_weight = entropy_weight  # 降低熵权重
        self.repulsion_weight = repulsion_weight  # 添加排斥权重
        self.temperature = temperature  # 添加温度参数
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, anchor_logits, pos_logits, neg_logits=None):
        # 正样本一致性 (与原ClusterLoss相同)
        anchor_normalized = F.normalize(anchor_logits.t(), p=2, dim=1)
        pos_normalized = F.normalize(pos_logits, p=2, dim=0)
        
        pos_similarity = torch.mm(anchor_normalized, pos_normalized) / self.temperature
        loss_ce = self.xentropy(pos_similarity, torch.arange(pos_similarity.size(0)).cuda())
        
        # 熵正则化 (权重降低)
        anchor_probs = self.softmax(anchor_logits)
        cluster_probs = anchor_probs.sum(0).view(-1)
        cluster_probs /= cluster_probs.sum()
        loss_ne = math.log(cluster_probs.size(0)) + (cluster_probs * torch.log(cluster_probs + 1e-10)).sum()
        
        # 簇间排斥项 (新增)
        loss_repel = 0.0
        if neg_logits is not None:
            neg_normalized = F.normalize(neg_logits, p=2, dim=0)
            # 最大化anchor和negative之间的距离
            neg_similarity = torch.mm(anchor_normalized, neg_normalized) / self.temperature
            # 理想情况下，负样本相似度应该接近0，所以最小化其平方和
            loss_repel = torch.mean(neg_similarity ** 2)
        
        # 总损失
        total_loss = loss_ce + self.entropy_weight * loss_ne + self.repulsion_weight * loss_repel
        
        return total_loss, loss_ce, loss_ne, loss_repel
    
     
class ClusterLoss(nn.Module):
    def __init__(self, entropy_weight=2.0):
        super(ClusterLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss().cuda()
        self.lamda = entropy_weight
        self.softmax = nn.Softmax(dim=1)
        self.temperature = 1.0

    def forward(self, ologits, plogits):
        """Partition Uncertainty Index

        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        assert ologits.shape == plogits.shape, ('Inputs are required to have same shape')

        ologits = self.softmax(ologits)
        plogits = self.softmax(plogits)

        # one-hot
        similarity = torch.mm(F.normalize(ologits.t(), p=2, dim=1), F.normalize(plogits, p=2, dim=0))
        loss_ce = self.xentropy(similarity, torch.arange(similarity.size(0)).cuda())

        # balance regularisation
        o = ologits.sum(0).view(-1)
        o /= o.sum()
        loss_ne = math.log(o.size(0)) + (o * o.log()).sum()

        loss = loss_ce + self.lamda * loss_ne

        return loss, loss_ce, loss_ne