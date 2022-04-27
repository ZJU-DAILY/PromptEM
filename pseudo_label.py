import copy
import logging
import math
import numpy as np
import torch
from scipy.cluster.hierarchy import to_tree, linkage
from sklearn.metrics import pairwise_distances, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from args import PromptEMArgs
from data import PromptEMData, Vocab, TypeDataset
from prompt import read_prompt_dataset, get_prompt_dataloader
from utils import enable_dropout, get_unique_label_trees


def test_pseudo_labels(args: PromptEMArgs, data: PromptEMData, model):
    labeled_dataset = read_prompt_dataset(data.left_entities, data.right_entities, data.train_pairs, data.train_y)
    labeled_dataloader = get_prompt_dataloader(args, labeled_dataset, shuffle=False)
    unlabeled_dataset = read_prompt_dataset(data.left_entities, data.right_entities, data.train_un_pairs, None)
    if len(unlabeled_dataset) == 0:
        # no unlabeled sample
        return 0
    unlabeled_dataloader = get_prompt_dataloader(args, unlabeled_dataset, shuffle=False)
    if args.test_pseudo_label == "uncertainty":
        ids, y_pred = gen_pseudo_labels_by_uncertainty(args, model, unlabeled_dataloader)
    elif args.test_pseudo_label == "confidence":
        ids, y_pred = gen_pseudo_labels_by_confidence(args, model, unlabeled_dataloader)
    elif args.test_pseudo_label == "fold_unfold":
        ids, y_pred = gen_pseudo_labels_by_fold_unfold(args, model, labeled_dataloader, unlabeled_dataloader)
    y_truth = [data.train_un_y[i] for i in ids]
    confusion = confusion_matrix(np.array(y_truth), np.array(y_pred), labels=[0, 1])
    FN = confusion[1][0]
    TN = confusion[0][0]
    TP = confusion[1][1]
    FP = confusion[0][1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    return TP, FN, TPR, TN, FP, TNR


def gen_pseudo_labels_by_confidence(args: PromptEMArgs, model, data_loader):
    model.eval()
    id = 0
    all_conf = []
    pos_all_conf = []
    neg_all_conf = []
    all_pred = []
    pos_all_pred = []
    neg_all_pred = []
    for batch in tqdm(data_loader):
        batch = batch.to(args.device)
        with torch.no_grad():
            logits = model(batch)
            logits = torch.softmax(logits, dim=-1)
            logits = logits.detach().cpu()
            for i, logit in enumerate(logits):
                if int(logit[1] > logit[0]) == 1:
                    pos_all_conf.append(abs(logit[0] - logit[1]))
                    pos_all_pred.append(1)
                else:
                    neg_all_conf.append(abs(logit[0] - logit[1]))
                    neg_all_pred.append(0)
                all_conf.append(abs(logit[0] - logit[1]))
                all_pred.append(int(logit[1] > logit[0]))
                id += 1
    if hasattr(data_loader, "raw_dataset"):
        k = math.floor(args.confidence_ratio * len(data_loader.raw_dataset))
    else:
        k = math.floor(args.confidence_ratio * len(data_loader.dataset))
    values, indices = torch.topk(-torch.tensor(all_conf), k=k)
    ids = indices.numpy().tolist()
    y_pred = torch.tensor(all_pred)[indices].numpy().tolist()
    return ids, y_pred


def gen_pseudo_labels_by_uncertainty(args: PromptEMArgs, model, data_loader):
    # MC Dropout
    model.eval()
    enable_dropout(model)
    all_std = []
    all_pred = []
    pos_all_std = []
    pos_all_ids = []
    pos_all_pred = []
    neg_all_std = []
    neg_all_ids = []
    neg_all_pred = []
    id = 0
    for batch in tqdm(data_loader):
        if hasattr(batch, "to"):
            batch = batch.to(args.device)
        else:
            x, labels = batch
            x = torch.tensor(x).to(args.device)
        with torch.no_grad():
            out_prob = []
            for _ in range(args.mc_dropout_pass):
                _batch = copy.deepcopy(batch)
                if hasattr(batch, "to"):
                    logits = model(_batch)
                else:
                    logits = model(x)
                logits = torch.softmax(logits, dim=-1)
                out_prob.append(logits.detach())
            out_prob = torch.stack(out_prob)
            out_std = torch.std(out_prob, dim=0)
            out_prob = torch.mean(out_prob, dim=0)
            max_value, max_idx = torch.max(out_prob, dim=1)
            max_std = out_std.gather(1, max_idx.view(-1, 1))
            max_std = max_std.squeeze(1).cpu().numpy().tolist()
            all_std.extend(max_std)
            all_pred.extend((out_prob[:, 1] > out_prob[:, 0]).type(torch.LongTensor).cpu().numpy().tolist())
            for i, std in enumerate(max_std):
                if int(out_prob[i][1] > out_prob[i][0]) == 1:
                    pos_all_std.append(std)
                    pos_all_pred.append(1)
                    pos_all_ids.append(id)
                else:
                    neg_all_std.append(std)
                    neg_all_pred.append(0)
                    neg_all_ids.append(id)
                id += 1
    if hasattr(data_loader, "raw_dataset"):
        k = math.ceil(args.uncertainty_ratio * len(data_loader.raw_dataset))
    else:
        k = math.ceil(args.uncertainty_ratio * len(data_loader.dataset))
    values, indices = torch.topk(-torch.tensor(all_std), k=k)
    ids = indices.numpy().tolist()
    y_pred = torch.tensor(all_pred)[indices].numpy().tolist()
    return ids, y_pred


def gen_pseudo_labels_by_fold_unfold(args: PromptEMArgs, model, labeled_dataloader, unlabeled_dataloader):
    """
        Modified by https://github.com/tdopierre/FewShotPseudoLabeling
    """
    model.eval()
    ids = []
    y_pred = []
    batch_features = []

    def hook(module, fea_in, fea_out):
        # print(fea_in[0].size())
        fea = torch.mean(fea_in[0], dim=1)
        batch_features.append(fea.detach().cpu())
        return None

    layer_name = 'prompt_model.plm.lm_head.decoder'
    handle = None
    for (name, module) in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook=hook)

    # all labeled embedding
    labeled_labels = []
    for batch in tqdm(labeled_dataloader):
        batch = batch.to(args.device)
        labeled_labels.extend(batch.label.cpu().numpy().tolist())
        with torch.no_grad():
            logits = model(batch)
    labeled_embeddings = torch.cat(batch_features, dim=0)
    batch_features.clear()
    # cluster every mini batch's unlabeled data
    for batch in tqdm(unlabeled_dataloader):
        batch = batch.to(args.device)
        with torch.no_grad():
            logits = model(batch)
            unlabeled_embeddings = torch.cat(batch_features, dim=0)
            embeddings = torch.cat((labeled_embeddings, unlabeled_embeddings), dim=0).numpy()
            unlabeled_labels = [-1 for _ in range(batch_features[0].size()[0])]
            labels = labeled_labels + unlabeled_labels
            batch_features.clear()
            labels_vocab = Vocab(labeled_labels)
            # sim matrix
            w = (1 - pairwise_distances(embeddings, embeddings, metric='cosine')).astype(np.float32)
            w_label = dict()
            for label in labels_vocab.labels:
                labelled_global_indices = [ix for ix, d in enumerate(labeled_labels) if d == label]
                w_label[label] = w[:, labelled_global_indices]
            # build a tree
            Z = linkage(embeddings, 'ward')
            root_tree = to_tree(Z)
            # split
            trees = get_unique_label_trees(root_tree=root_tree, labels=labels)
            # generate pseudo-labels
            for tree, path in trees:
                order = tree.pre_order()
                tree_labels = [labels[ix] for ix in order]
                if set(tree_labels) == {''}:
                    continue
                pseudo_labels = [l for o, l in zip(order, tree_labels) if labels[o] != -1]
                if len(pseudo_labels) > 0:
                    pseudo_label = pseudo_labels[0]
                    for ix in order:
                        if labels[ix] == -1:
                            ids.append(ix - len(labeled_labels))
                            y_pred.append(pseudo_label)
    handle.remove()
    return ids, y_pred


def gen_pseudo_labels(args: PromptEMArgs, data: PromptEMData, model, prompt=True) -> int:
    """
        update data.train_pairs, data.train_y, data.train_un_pairs
        return the number of the generated pseudo-labels
    """
    if prompt:
        labeled_dataset = read_prompt_dataset(data.left_entities, data.right_entities, data.train_pairs, data.train_y)
        labeled_dataloader = get_prompt_dataloader(args, labeled_dataset, shuffle=False)
        unlabeled_dataset = read_prompt_dataset(data.left_entities, data.right_entities, data.train_un_pairs, None)
        if len(unlabeled_dataset) == 0:
            # no unlabeled sample
            return 0
        unlabeled_dataloader = get_prompt_dataloader(args, unlabeled_dataset, shuffle=False)
    else:
        labeled_dataset = TypeDataset(data, "train")
        unlabeled_dataset = TypeDataset(data, "un")
        labeled_dataloader = DataLoader(dataset=labeled_dataset, batch_size=args.batch_size,
                                        collate_fn=TypeDataset.pad, )
        unlabeled_dataloader = DataLoader(dataset=unlabeled_dataset, batch_size=args.batch_size,
                                          collate_fn=TypeDataset.pad)
    inter_samples = None
    methods = args.pseudo_label_method.split("+")
    logging.info(methods)
    for met in methods:
        if met == "confidence":
            ids, y_pred = gen_pseudo_labels_by_confidence(args, model, unlabeled_dataloader)
        elif met == "uncertainty":
            ids, y_pred = gen_pseudo_labels_by_uncertainty(args, model, unlabeled_dataloader)
        elif met == "fold_unfold":
            ids, y_pred = gen_pseudo_labels_by_fold_unfold(args, model, labeled_dataloader, unlabeled_dataloader)
        if inter_samples is None:
            inter_samples = set([(x, y) for x, y in zip(ids, y_pred)])
        else:
            tmp_ids = set([(x, y) for x, y in zip(ids, y_pred)])
            inter_samples = inter_samples.intersection(tmp_ids)

    inter_ids = [x[0] for x in list(inter_samples)]
    inter_y = [x[1] for x in list(inter_samples)]
    new_train_un_pairs = []
    for i, pair in enumerate(data.train_un_pairs):
        if i in inter_ids:
            data.train_pairs.append(pair)
            data.train_y.append(inter_y[inter_ids.index(i)])
        else:
            new_train_un_pairs.append(pair)
    data.train_un_pairs = new_train_un_pairs
    return len(inter_ids)
