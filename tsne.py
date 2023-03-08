import numpy as np
import torch
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn import manifold
from transformers import AlbertTokenizer
from matplotlib import pyplot as plt
from dataset import load_data
from model import LogContrast


if __name__ == '__main__':
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader, test_dataloader = load_data(log_type='HDFS',
                                                  train_data_dir='./output/HDFS/HDFS_train_10000.csv',
                                                  test_data_dir='./output/HDFS/HDFS_test_575061.csv',
                                                  tokenizer=tokenizer,
                                                  train_batch_size=50,
                                                  test_batch_size=50)

    logcontrast = LogContrast(vocab_size=120,
                              feat_dim=512,
                              feat_type='logkey',
                              semantic_model_name='albert')
    logcontrast.to(device)

    logcontrast.load_state_dict(torch.load('output/HDFS/model_contrastive_epoch20.pt'))

    all_features, all_labels = [], []

    logcontrast.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(train_dataloader, desc='Evaluating')):
            semantics = {
                'input_ids': batch['semantics']['input_ids'].to(device),
                'token_type_ids': batch['semantics']['token_type_ids'].to(device),
                'attention_mask': batch['semantics']['attention_mask'].to(device),
            }
            sequences = batch['sequences'].to(device)
            seqence_masks = batch['sequence_masks'].to(device)
            labels = batch['labels'].to(device)
            logits, feature = logcontrast(semantics, sequences, seqence_masks)

            all_features.append(feature)
            all_labels.append(labels)

    all_features = torch.vstack(all_features)
    all_labels = torch.vstack(all_labels)
    all_features = all_features.detach().cpu().numpy()
    all_labels = all_labels.view(-1).detach().cpu().numpy()
    print(all_features.shape, all_labels.shape, all_labels[107])

    num_classes = len(np.unique(all_labels))

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    tsne_feature = tsne.fit_transform(all_features)

    df = pd.DataFrame()
    df['label'] = all_labels
    df['f1'] = tsne_feature[:, 0]
    df['f2'] = tsne_feature[:, 1]
    sns.scatterplot(x='f1', y='f2', hue=df['label'].tolist(),
                    palette=sns.color_palette('hls', num_classes), data=df).set(title='t-SNE')
    plt.show()

