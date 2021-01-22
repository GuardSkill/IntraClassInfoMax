import torch
import torch.nn.functional as F
from models.resencoder import resnet18
from models.resencoder import GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator
import torch.nn as nn
import numpy as np

class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=1):
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

    def prior_loss(self, y):
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = -(term_a + term_b) * self.gamma
        return PRIOR

    def forward(self, y, M, M_prime):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand((-1, -1, M.shape[2], M.shape[3]))

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        return LOCAL + GLOBAL


class GlobalD(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(128, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 24 * 24 + 512, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalD(nn.Module):
    def __init__(self):
        super().__init__()
        # self.c0 = nn.Conv2d(192, 512, kernel_size=1)
        self.c0 = nn.Conv2d(640, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorD(nn.Module):
    def __init__(self):
        super().__init__()
        # self.l0 = nn.Linear(64, 1000)
        # self.l1 = nn.Linear(1000, 200)
        # self.l2 = nn.Linear(200, 1)
        self.l0 = nn.Linear(512, 640)
        self.l1 = nn.Linear(640, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class IntraClsInfoMax(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=1):
        super().__init__()
        self.global_d = GlobalD()
        self.local_d = LocalD()
        self.prior_d = PriorD()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def prior_loss(self, y):
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = -(term_a + term_b) * self.gamma
        return PRIOR

    def forward(self, y, M, M_prime,lablels):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        label,label_prime=lablels
        # not intra class
        out_idx = torch.where(label != label_prime)

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand((-1, -1, M.shape[2], M.shape[3]))

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime[out_idx], y_exp[out_idx]), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y[out_idx], M_prime[out_idx])).mean()
        GLOBAL = (Em - Ej) * self.alpha

        return LOCAL + GLOBAL


if __name__ == '__main__':
    N, D_in, H, D_out = 64, 3, 100, 10
    B = 128
    H = 208
    W = H
    dtype = torch.float
    device = torch.device("cuda")
    # x=torch.randn(123,D_in, H, device=device, dtype=dtype, requires_grad=True)
    x = torch.zeros(B, D_in, H, W, device=device, dtype=dtype, requires_grad=True)
    encoder = resnet18(True).cuda()
    y, M = encoder(x, True)
    DIM_loss_fn = DeepInfoMaxLoss().to(device)
    # rotate images to create pairs for comparison
    print(M[1:].shape)
    print(M[0].unsqueeze(0).shape)
    M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)  # move to front one by one
    loss = DIM_loss_fn(y, M, M_prime)
