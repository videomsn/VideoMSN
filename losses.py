import itertools

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.distributed as dist

class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads

class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads

def init_msn_loss(
    num_views=1,
    tau=0.1,
    me_max=True,
    return_preds=False
):
    """
    Make unsupervised MSN loss

    :num_views: number of anchor views
    :param tau: cosine similarity temperature
    :param me_max: whether to perform me-max regularization
    :param return_preds: whether to return anchor predictions
    """
    softmax = torch.nn.Softmax(dim=1)

    def sharpen(p, T):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(query, supports, support_labels, temp=tau):
        """ Soft Nearest Neighbours similarity classifier """
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)
        return softmax(query @ supports.T / temp) @ support_labels

    def loss(
        anchor_views,
        target_views,
        prototypes,
        proto_labels,
        T=0.25,
        use_entropy=False,
        use_sinkhorn=False,
        sharpen=sharpen,
        snn=snn
    ):
        # Step 1: compute anchor predictions
        probs = snn(anchor_views, prototypes, proto_labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = sharpen(snn(target_views, prototypes, proto_labels), T=T)
            if use_sinkhorn:
                targets = distributed_sinkhorn(targets)
            targets = torch.cat([targets for _ in range(num_views)], dim=0)

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            # avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
            avg_probs = torch.mean(probs, dim=0)
            rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            # print("me_max active")

        sloss = 0.
        if use_entropy:
            sloss = torch.mean(torch.sum(torch.log(probs**(-probs)), dim=1))

        # -- logging
        with torch.no_grad():
            num_ps = float(len(set(targets.argmax(dim=1).tolist())))
            max_t = targets.max(dim=1).values.mean()
            min_t = targets.min(dim=1).values.mean()
            log_dct = {'np': num_ps, 'max_t': max_t, 'min_t': min_t}

        if return_preds:
            return loss, rloss, sloss, log_dct, targets

        return loss, rloss, sloss, log_dct

    return loss


@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True):
    _got_dist = use_dist and torch.distributed.is_available() \
        and torch.distributed.is_initialized() \
        and (torch.distributed.get_world_size() > 1)

    if _got_dist:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    Q = Q.T
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T

class DeepMutualLoss(nn.Module):

    def __init__(self, base_criterion, w, temperature=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.w = w if w > 0  else -w
        self.T = temperature

        self.neg = w < 0

    def forward(self, logits, targets):
        n = len(logits)

        # CE losses
        ce_loss = [self.base_criterion(logits[i], targets) for i in range(n)]
        ce_loss = torch.sum(torch.stack(ce_loss, dim=0), dim=0)

        # KD Loss
        kd_loss = [1. / (n-1) * 
                   self.kd_criterion(
                       F.log_softmax(logits[i] / self.T, dim=1), 
                       F.log_softmax(logits[j] / self.T, dim=1).detach()
                   ) * self.T * self.T
                   for i, j in itertools.permutations(range(n), 2)]
        kd_loss = torch.sum(torch.stack(kd_loss, dim=0), dim=0)
        if self.neg:
            kd_loss = -1.0 * kd_loss

        total_loss = (1.0 - self.w) * ce_loss + self.w * kd_loss
        return total_loss, ce_loss.detach(), kd_loss.detach()


class ONELoss(nn.Module):

    def __init__(self, base_criterion, w, temperature=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.w = w
        self.T = temperature

    def forward(self, logits, targets):
        n = len(logits)
        ensemble_logits = torch.mean(torch.stack(logits, dim=0), dim=0)

        # CE losses
        ce_loss = [self.base_criterion(logits[i], targets) for i in range(n)] + [self.base_criterion(ensemble_logits, targets)]
        #ce_loss = torch.sum(torch.stack(ce_loss, dim=0), dim=0)
        ce_loss = torch.mean(torch.stack(ce_loss, dim=0), dim=0)

        # One Loss
        kd_loss = [self.kd_criterion(
            F.log_softmax(logits[i] / self.T, dim=1), 
            F.log_softmax(ensemble_logits / self.T, dim=1).detach()
        ) * self.T * self.T for i in range(n)]
        #kd_loss = torch.sum(torch.stack(kd_loss, dim=0), dim=0)
        kd_loss = torch.mean(torch.stack(kd_loss, dim=0), dim=0)

        #total_loss = (1.0 - self.w) * ce_loss + self.w * kd_loss
        #total_loss = (1.0 - self.w) * ce_loss - self.w * kd_loss
        total_loss = ce_loss + self.w * kd_loss
        return total_loss, ce_loss.detach(), kd_loss.detach()


class MulMixLabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(MulMixLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target, beta=1.0):
        inv_prob = torch.pow(1.0 - F.softmax(x, dim=-1), beta)
        logprobs = F.log_softmax(x, dim=-1)
        logprobs = logprobs * inv_prob
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MulMixSoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(MulMixSoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target, beta=1.0):
        inv_prob = torch.pow(1.0 - F.softmax(x, dim=-1), beta)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1) * inv_prob, dim=-1)
        return loss.mean()


class MulMixturelLoss(nn.Module):

    def __init__(self, base_criterion, beta):
        super().__init__()

        if isinstance(base_criterion, LabelSmoothingCrossEntropy):
            self.base_criterion = MulMixLabelSmoothingCrossEntropy(base_criterion.smoothing)
        elif isinstance(base_criterion, SoftTargetCrossEntropy):
            self.base_criterion = MulMixSoftTargetCrossEntropy()
        else:
            raise ValueError("Unknown type")
            
        self.beta = beta

    def forward(self, logits, targets):
        n = len(logits)

        # CE losses
        ce_loss = [self.base_criterion(logits[i], targets, self.beta / (n - 1)) for i in range(n)]
        ce_loss = torch.sum(torch.stack(ce_loss, dim=0), dim=0)
        return ce_loss


class SelfDistillationLoss(nn.Module):

    def __init__(self, base_criterion, w, temperature=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.w = w
        self.T = temperature

    def forward(self, logits, targets):
        # logits is a list, the first one is the reference logits for self-distillation

        # CE losses
        ce_loss = self.base_criterion(logits[1], targets)

        # KD Loss
        kd_loss = self.kd_criterion(
            F.log_softmax(logits[1] / self.T, dim=1),
            F.log_softmax(logits[0] / self.T, dim=1).detach()
        ) * self.T * self.T

        total_loss = (1.0 - self.w) * ce_loss + self.w * kd_loss
        return total_loss, ce_loss.detach(), kd_loss.detach()
