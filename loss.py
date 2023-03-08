import torch
from torch import nn


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, logits, label, seq_feature=None):
        loss = self.xent_loss(logits, label)
        return {
            'loss': loss,
            'cross_entropy_loss': loss
        }


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.05, lambda_c=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.xent_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.lambda_c = lambda_c

    def forward(self, logits, label, seq_feature):
        batch_size = seq_feature.shape[0]

        z1 = self.dropout1(seq_feature)
        z2 = self.dropout2(seq_feature)

        nz1 = z1.norm(dim=1)
        nz2 = z2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / torch.einsum('i,j->ij', nz1, nz2)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        contrastive_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        contrastive_loss = -torch.log(contrastive_loss).mean()

        ce_loss = self.xent_loss(logits, label)

        loss = ce_loss + self.lambda_c * contrastive_loss

        return {
            'loss': loss,
            'cross_entropy_loss': ce_loss,
            'contrastive_loss': contrastive_loss
        }
