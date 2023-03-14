import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from config import load_configs
from dataset import load_data
from loss import CELoss, CLLoss
from model import LogContrast
from utils import set_seed


def train(model, dataloader, criterion, optimizer, device, sup_ratio=1.0):
    losses = defaultdict(list)
    all_predicts, all_labels = [], []
    model.train()
    for batch in tqdm(dataloader, desc='Training'):
        semantics = {
            'input_ids': batch['semantics']['input_ids'].to(device),
            'token_type_ids': batch['semantics']['token_type_ids'].to(device),
            'attention_mask': batch['semantics']['attention_mask'].to(device),
        }
        sequences = batch['sequences'].to(device)
        seqence_masks = batch['sequence_masks'].to(device)
        labels = batch['labels'].to(device)

        logits, feature = model(semantics, sequences, seqence_masks)
        predicts = torch.argmax(logits, dim=1)

        if isinstance(criterion, CELoss):
            all_losses = criterion(logits, labels)
        elif isinstance(criterion, CLLoss):
            # unsupervised learning
            if np.random.rand() < sup_ratio:
                all_losses = criterion(logits, labels, feature, True)
            else:
                all_losses = criterion(logits, labels, feature, False)
        else:
            raise ValueError('`criterion` must be ["ce_loss", "cl_loss"]')

        loss = all_losses['loss']
        losses['loss'].append(loss.item())

        if 'ce_loss' in all_losses.keys():
            losses['ce_loss'].append(all_losses['ce_loss'].item())

        if 'cl_loss' in all_losses.keys():
            losses['cl_loss'].append(all_losses['cl_loss'].item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_predicts += predicts.detach().cpu().numpy().tolist()
        all_labels += labels.detach().cpu().numpy().tolist()

    return {
        'loss': np.mean(losses['loss']),
        'ce_loss': np.mean(losses['ce_loss']) if 'ce_loss' in losses.keys() else None,
        'cl_loss': np.mean(losses['cl_loss']) if 'cl_loss' in losses.keys() else None,
        'precision': precision_score(all_labels, all_predicts),
        'recall': recall_score(all_labels, all_predicts),
        'f1': f1_score(all_labels, all_predicts),
        'acc': accuracy_score(all_labels, all_predicts)
    }


def test(model, dataloader, criterion, device):
    losses = defaultdict(list)
    all_predicts, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            semantics = {
                'input_ids': batch['semantics']['input_ids'].to(device),
                'token_type_ids': batch['semantics']['token_type_ids'].to(device),
                'attention_mask': batch['semantics']['attention_mask'].to(device),
            }
            sequences = batch['sequences'].to(device)
            seqence_masks = batch['sequence_masks'].to(device)
            labels = batch['labels'].to(device)

            logits, feature = model(semantics, sequences, seqence_masks)
            predicts = torch.argmax(logits, dim=1)

            all_losses = criterion(logits, labels, feature)
            loss = all_losses['loss']
            losses['loss'].append(loss.item())

            if 'ce_loss' in all_losses.keys():
                losses['ce_loss'].append(all_losses['ce_loss'].item())

            if 'cl_loss' in all_losses.keys():
                losses['cl_loss'].append(all_losses['cl_loss'].item())

            all_predicts += predicts.detach().cpu().numpy().tolist()
            all_labels += labels.detach().cpu().numpy().tolist()

    return {
        'loss': np.mean(losses['loss']),
        'ce_loss': np.mean(losses['ce_loss']) if 'ce_loss' in losses.keys() else None,
        'cl_loss': np.mean(losses['cl_loss']) if 'cl_loss' in losses.keys() else None,
        'precision': precision_score(all_labels, all_predicts),
        'recall': recall_score(all_labels, all_predicts),
        'f1': f1_score(all_labels, all_predicts),
        'acc': accuracy_score(all_labels, all_predicts)
    }


if __name__ == '__main__':
    args, logger = load_configs()

    assert args.do_train or args.do_eval, '`do_train` and `do_test` should be at least true for one'

    logger.info(f'Parameters: {args}')

    set_seed(args.seed)

    if args.semantic_model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.semantic_model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
    elif args.semantic_model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    else:
        raise ValueError('`semantic_model_name` must be in ["bert", "roberta", "albert"]')

    train_dataloader, test_dataloader = load_data(log_type=args.log_type,
                                                  train_data_dir=args.train_data_dir,
                                                  test_data_dir=args.test_data_dir,
                                                  tokenizer=tokenizer,
                                                  train_batch_size=args.train_batch_size,
                                                  test_batch_size=args.test_batch_size)

    logcontrast = LogContrast(vocab_size=args.vocab_size,
                              feat_dim=args.feat_dim,
                              feat_type=args.feat_type,
                              semantic_model_name=args.semantic_model_name)
    logcontrast.to(args.device)

    params = filter(lambda p: p.requires_grad, logcontrast.parameters())
    num_params = sum([p.nelement() for p in params])
    logger.info(f'Number of trainable parameters: {num_params}')

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.loss_fct == 'ce':
        criterion = CELoss()
    elif args.loss_fct == 'cl':
        criterion = CLLoss(temperature=args.temperature, lambda_cl=args.lambda_cl)
    else:
        raise ValueError('`loss_fct` must be in ["ce", "cl"]')

    save_dir = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    if args.do_train:
        best_results = {'loss': np.Inf, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'acc': 0.0}
        best_metric_to_save = 'acc'
        logger.info('Training start:')
        for epoch in tqdm(range(args.num_epoch), desc='Epoch'):
            train_results = train(logcontrast, train_dataloader, criterion, optimizer, args.device, args.sup_ratio)
            logger.info('[train]')
            logger.info(f'epoch: {epoch + 1}/{args.num_epoch} - {100 * (epoch + 1) / args.num_epoch:.2f}%')
            logger.info(f'loss: {train_results["loss"]}')
            logger.info(f'ce_loss: {train_results["ce_loss"] if train_results["ce_loss"] else None}')
            logger.info(f'cl_loss: {train_results["cl_loss"] if train_results["cl_loss"] else None}')
            logger.info(f'precision: {100 * train_results["precision"]}')
            logger.info(f'recall: {100 * train_results["recall"]}')
            logger.info(f'f1: {100 * train_results["f1"]}')
            logger.info(f'acc: {100 * train_results["acc"]}')
            if best_results[best_metric_to_save] < train_results[best_metric_to_save]:
                best_results.update(train_results)
                torch.save(logcontrast.state_dict(), save_dir)
                logger.info(f'model saved at "{save_dir}"')

    if args.do_test:
        logcontrast.load_state_dict(torch.load(save_dir))
        logger.info('Testing start:')
        test_results = test(logcontrast, test_dataloader, criterion, args.device)
        logger.info('[test]')
        logger.info(f'loss: {test_results["loss"]}')
        logger.info(f'ce_loss: {test_results["ce_loss"] if test_results["ce_loss"] else None}')
        logger.info(f'cl_loss: {test_results["cl_loss"] if test_results["cl_loss"] else None}')
        logger.info(f'precision: {100 * test_results["precision"]}')
        logger.info(f'recall: {100 * test_results["recall"]}')
        logger.info(f'f1: {100 * test_results["f1"]}')
        logger.info(f'acc: {100 * test_results["acc"]}')

    logger.info(f'log saved at: "{args.log_name}"')
