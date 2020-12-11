from torch.nn.modules import Module
import torch.nn._reduction as _Reduction
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class HHLoss(_Loss):
    r"""Creates a criterion that measures the hierarchy label error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(HHLoss, self).__init__(size_average, reduce, reduction)
        self.norm_p = 2
        self.param_lambda = 0.001
        self.param_beta = 0.002

    def forward(self, input, target):
        input_s = input.mm(input.t())
        target_s = target.mm(target.t())
        uncorr_M = input_s.mm(input_s.t())/input_s.shape[0]
        I = torch.eye(input_s.shape[1]).type_as(uncorr_M)
        loss = torch.norm(input_s-target_s,p=self.norm_p)\
               + self.param_lambda*torch.norm(torch.sum(input_s,0),p=self.norm_p)\
               + self.param_beta*torch.norm(uncorr_M-I,p=self.norm_p)
        return loss
        # input_s = input.mm(input.t())
        # target_s = target.mm(target.t())
        # return F.mse_loss(input_s, target_s, reduction=self.reduction)

class HHLoss_bin(_Loss):
    r"""Creates a criterion that measures the hierarchy label error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(HHLoss_bin, self).__init__(size_average, reduce, reduction)
        self.th = 0.0
        self.norm_p = 2
        self.param_lambda = 0.001
        self.param_beta = 0.002
        self.mu = 0.001

    def forward(self, input, target):
        input_b = torch.sign(input)
        input_s = input.mm(input.t())
        # input_s = input_b.mm(input_b.t())
        target_s = target.mm(target.t())
        uncorr_M = input_s.mm(input_s.t()) / input_s.shape[0]
        I = torch.eye(input_s.shape[1]).type_as(uncorr_M)
        loss = torch.norm(input_s - target_s, p=self.norm_p) \
               + self.param_lambda * torch.norm(torch.sum(input, 0), p=self.norm_p) \
               + self.param_beta * torch.norm(uncorr_M - I, p=self.norm_p) \
               + self.mu*torch.norm(torch.sum(input_b-input, 0), p=self.norm_p)
        return loss
        # input = (torch.sign(input)+1)/2
        # input = torch.sign(input)
        # input_s = input.mm(input.t())
        # target_s = target.mm(target.t())
        # uncorr_M = input_s.mm(input_s.t()) / input_s.shape[0]
        # I = torch.eye(input_s.shape[1]).type_as(uncorr_M)
        # loss = torch.norm(input_s - target_s, p=self.norm_p) \
        #        + self.param_lambda * torch.norm(torch.sum(input, 0), p=self.norm_p) \
        #        + self.param_beta * torch.norm(uncorr_M - I, p=self.norm_p) \
        #        + self.mu*torch.norm(torch.sum(input, 0)
        # return loss

class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class BalancedLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, device='cpu'):
        super(BalancedLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        self.alpha = torch.histc(ids, bins=self.class_num, min=0, max=self.class_num-1).float()/float(ids.shape[0])
        # self.alpha = 1.0*self.alpha.reciprocal() # 10, 100
        self.alpha = 1.0 - self.alpha/10.0
        alpha_c = self.alpha[ids.data.view(-1)]
        if inputs.is_cuda and not alpha_c.is_cuda:
            alpha_c = alpha_c.to(self.device)
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha_c*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class CosineLoss(nn.Module):
    r"""
        This criterion is a implemenation of Cosine Loss
    """
    def __init__(self, size_average=True):
        super(CosineLoss, self).__init__()

        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = torch.div(inputs,inputs.norm(dim=1,keepdim=True))
        # one-hot coding
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = 1.0-probs

        if self.size_average:
            loss = log_p.mean()
        else:
            loss = log_p.sum()
        return loss