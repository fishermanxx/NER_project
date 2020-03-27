import torch
from torch import nn
import copy

class EMA():
    def __init__(self, model, mu=0.999):
        self.model = model
        self.mu = mu
        self.shadow = {}
        self.backup = {}
        self.old_ema = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def backup_oldema(self):
        self.old_ema = copy.deepcopy(self.shadow)

    def return_oldema(self):
        self.shadow = copy.deepcopy(self.old_ema)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(2).requires_grad_()
            # self.alpha = torch.ones(2, requires_grad=False)  ##no grad
        else:
            assert alpha.shape == (2,), 'this is for sigmoid, alpha dim must be (2,)'
            if alpha.requires_grad:
                # alpha.requires_grad=False  ##no grad  
                self.alpah = alpha.float()
            else:
                self.alpha = alpha.float().requires_grad_()
                # self.alpha = alpha.float()  ##no grad
        
        self.gamma = gamma
        self.class_num = 2
        self.size_average = size_average

    def forward(self, inputs, targets):
        '''
        :param
            @inputs: (N1, N2, ..., Nn), torch.tensor  ##(经过sigmoid之后的值)
            @targets: (N1, N2, ..., Nn), torch.tensor  ##(如果是softmax, target是一串数列, [0, 3, 2, 2, 3]这种，可以使用torch.scatter_)
            @ref: https://zhuanlan.zhihu.com/p/28527749 for softmax
        '''
        if inputs.is_cuda:
            self.alpha = self.alpha.cuda()

        y1loss = -self.alpha[1]*inputs.log()*targets
        weight1 = (1-inputs)**self.gamma
        
        y0loss = -self.alpha[0]* ((1-inputs).log()) *(1-targets)
        weight0 = inputs**self.gamma
        
        loss = y1loss*weight1 + y0loss*weight0
        
        if self.size_average:
            return loss.mean()
        else:
            return loss
        
           
if __name__ == '__main__':
    a = torch.tensor([1, 10]).float()

    inputs = torch.randn((1, 5, 2))
    inputs = torch.sigmoid(inputs)
    targets = torch.randint(low=0, high=2, size=(1, 5, 2)).float()

    weight_a = a[targets.long()]

    FCloss = FocalLoss(alpha=a, gamma=0, size_average=False)
    BCEloss = nn.BCELoss(weight=weight_a, reduction='none')

    fc = FCloss(inputs, targets)
    check = BCEloss(inputs, targets)
    print(fc)
    print(check)
    print(fc-check)