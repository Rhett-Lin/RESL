import torch
import torch.nn.functional as F
from torch import nn
import sys


def entropy(logit):
    logit = logit.mean(dim=0)
    logit_ = torch.clamp(logit, min=1e-9)
    b = logit_ * torch.log(logit_)
    return -b.sum()


def consistency_loss(anchors, neighbors):
    b, n = anchors.size()
    similarity = torch.bmm(anchors.view(b, 1, n), neighbors.view(b, n, 1)).squeeze()
    ones = torch.ones_like(similarity)
    consistency_loss = F.binary_cross_entropy(similarity, ones)

    return consistency_loss


class DistillLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(DistillLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.mask = self.mask_correlated_clusters(class_num).cuda()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        c = F.normalize(c, dim=1)
        sim = c @ c.T / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss
    


class ContrastiveInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, c_i, c_j):
        batch_size = c_i.size(0)
        
        c_i = F.normalize(c_i, dim=1)
        c_j = F.normalize(c_j, dim=1)

        positive_similarity = torch.einsum('ij,ij->i', c_i, c_j) / self.temperature
        negative_similarity = torch.mm(c_i, c_j.t()) / self.temperature

        labels = torch.arange(batch_size).long().to(c_i.device)

        logits = negative_similarity
        logits.fill_diagonal_(-float('inf'))

        logits = torch.cat([positive_similarity.unsqueeze(1), logits], dim=1)

        loss = self.criterion(logits, labels)

        return loss
    
    
EPS = sys.float_info.epsilon  

def mutual_information(x_img, x_txt):
    _, k = x_img.size()
    p_i_j = compute_joint(x_img, x_txt)
    assert (p_i_j.size() == (k, k))

    temp1 = p_i_j.sum(dim=1).view(k, 1)
    p_i = temp1.expand(k, k).clone()
    temp2 = p_i_j.sum(dim=0).view(1, k)
    p_j = temp2.expand(k, k).clone()

    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    loss = loss.sum()
    return loss

def compute_joint(x_img, x_txt):
    bn, k = x_img.size()
    assert (x_txt.size(0) == bn and x_txt.size(1) == k)

    p_i_j = x_img.unsqueeze(2) * x_txt.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.
    p_i_j = p_i_j / p_i_j.sum()
    return p_i_j

