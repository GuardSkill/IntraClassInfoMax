import torch
import torch.nn.functional as F
from models.resencoder import resnet18
from models.resencoder import GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator
import torch.nn as nn


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, y, M, M_prime):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand((-1, -1, 28, 28))

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR
if __name__ == '__main__':
    N, D_in, H, D_out = 64, 3, 100, 10
    B=128
    H=208
    W=H
    dtype = torch.float
    device = torch.device("cuda")
    # x=torch.randn(123,D_in, H, device=device, dtype=dtype, requires_grad=True)
    x = torch.zeros(B,D_in, H, W,device=device, dtype=dtype, requires_grad=True)
    encoder=resnet18(True).cuda()
    y, M = encoder(x,True)
    DIM_loss_fn = DeepInfoMaxLoss().to(device)
    # rotate images to create pairs for comparison
    print(M[1:].shape)
    print(M[0].unsqueeze(0).shape)
    M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)    # move to front one by one
    loss = DIM_loss_fn (y, M, M_prime)