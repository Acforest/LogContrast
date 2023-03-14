import torch
import random
import numpy as np


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lcs(seq1, seq2):
    len1, len2 = len(seq1), len(seq2)
    dp = [[0 for i in range(len2 + 1)] for j in range(len1 + 1)]
    for i in range(len1):
        for j in range(len2):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            elif dp[i + 1][j] > dp[i][j + 1]:
                dp[i + 1][j + 1] = dp[i + 1][j]
            else:
                dp[i + 1][j + 1] = dp[i][j + 1]
    return dp[-1][-1]


if __name__ == '__main__':
    seq1 = torch.tensor([1, 2, 3, 1, 0, 0, 0, 0])
    seq2 = torch.tensor([1, 1, 2, 3, 0, 0, 0, 0])
    seq1 = list(filter(lambda x: x != 0, seq1))
    seq2 = list(filter(lambda x: x != 0, seq2))
    print(seq1, seq2)
    print(lcs(seq1, seq2))
