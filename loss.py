import torch
from torch import nn


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, logits, label):
        loss = self.xent_loss(logits, label)
        return {
            'loss': loss,
            'ce_loss': loss
        }


class CLLoss(nn.Module):

    def __init__(self, temperature=0.5, lambda_cl=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.xent_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.lambda_cl = lambda_cl

    def forward(self, logits, label, seq_feature, supervised=True):
        batch_size = seq_feature.shape[0]

        z1 = self.dropout1(seq_feature)
        z2 = self.dropout2(seq_feature)

        nz1 = z1.norm(dim=1)
        nz2 = z2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / torch.einsum('i,j->ij', nz1, nz2)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        cl_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        cl_loss = -torch.log(cl_loss).mean()

        ce_loss = self.xent_loss(logits, label)

        if supervised:
            loss = ce_loss + self.lambda_cl * cl_loss
        else:
            loss = self.lambda_cl * cl_loss

        return {
            'loss': loss,
            'ce_loss': ce_loss,
            'cl_loss': cl_loss
        }
