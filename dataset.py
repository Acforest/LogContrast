from functools import partial
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class LogDataset(Dataset):
    def __init__(self,
                 raw_data,
                 max_seq_len: int = 32):
        self.dataset = []
        dataset_len = len(raw_data['semantics'])
        for idx in range(dataset_len):
            log_semantics = raw_data['semantics'][idx].split()
            log_sequence = raw_data['sequences'][idx].strip('[').rstrip(']').split(',')
            log_sequence = [int(logkey) for logkey in log_sequence]
            log_sequence_mask = [1] * len(log_sequence)
            if len(log_sequence) < max_seq_len:
                log_sequence += [0] * (max_seq_len - len(log_sequence))
                log_sequence_mask += [0] * (max_seq_len - len(log_sequence_mask))
            else:
                log_sequence = log_sequence[:max_seq_len]
                log_sequence_mask = log_sequence_mask[:max_seq_len]
            label = 0 if raw_data['labels'][idx] == 'Normal' else 1
            self.dataset.append((log_semantics, log_sequence, log_sequence_mask, label))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch, tokenizer):
    log_semantics, log_sequences, log_sequence_masks, labels = map(list, zip(*batch))
    log_semantics = tokenizer(log_semantics,
                              padding=True,
                              truncation=True,
                              max_length=256,
                              is_split_into_words=True,
                              add_special_tokens=True,
                              return_tensors='pt')
    log_sequences = torch.tensor(log_sequences, dtype=torch.long)
    log_sequence_masks = torch.tensor(log_sequence_masks, dtype=torch.bool)
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        'semantics': log_semantics,
        'sequences': log_sequences,
        'sequence_masks': log_sequence_masks,
        'labels': labels
    }


def load_data(log_type: str,
              train_data_dir: str,
              test_data_dir: str,
              tokenizer,
              train_batch_size: int,
              test_batch_size: int):
    train_data_df = pd.read_csv(train_data_dir)
    test_data_df = pd.read_csv(test_data_dir)
    train_raw_data = {
        'semantics': train_data_df['EventTemplateSequence'].values.tolist(),
        'sequences': train_data_df['EventIdSequence'].values.tolist(),
        'labels': train_data_df['Label'].values.tolist()
    }
    test_raw_data = {
        'semantics': test_data_df['EventTemplateSequence'].values.tolist(),
        'sequences': test_data_df['EventIdSequence'].values.tolist(),
        'labels': test_data_df['Label'].values.tolist()
    }
    train_dataset = LogDataset(train_raw_data)
    test_dataset = LogDataset(test_raw_data)
    log_collate_fn = partial(collate_fn, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, train_batch_size, pin_memory=True, collate_fn=log_collate_fn)
    test_dataloader = DataLoader(test_dataset, test_batch_size, pin_memory=True, collate_fn=log_collate_fn)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataloader, test_dataloader = load_data('HDFS',
                                                  train_data_dir=r'E:\LogX\output\HDFS\HDFS_train_10000.csv',
                                                  test_data_dir=r'E:\LogX\output\HDFS\HDFS_test_575061.csv',
                                                  tokenizer=tokenizer,
                                                  train_batch_size=64,
                                                  test_batch_size=64)
    for batch in train_dataloader:
        print(batch)
        break
