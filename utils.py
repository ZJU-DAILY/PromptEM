import json
from datetime import datetime
import logging
import os
import random
from typing import List
from scipy import cluster
from torch.nn.functional import one_hot
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data import PromptEMData
from args import PromptEMArgs
from summarize import Summarizer
from openprompt import PromptForClassification
import csv


def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)


def set_logger(name):
    cur_time = '_' + datetime.now().strftime('%F %T')
    name += cur_time
    name = name.replace(":", "_")
    """
    Write logs to checkpoints and console.
    """
    log_file = os.path.join('./logs', name)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def evaluate(y_truth, y_pred, return_acc=False):
    precision = precision_score(y_truth, y_pred, zero_division=0)
    recall = recall_score(y_truth, y_pred, zero_division=0)
    f1 = f1_score(y_truth, y_pred, zero_division=0)
    acc = accuracy_score(y_truth, y_pred)
    if return_acc:
        return precision, recall, f1, acc
    else:
        return precision, recall, f1


def read_ground_truth(file_path, files=None):
    if files is None:
        files = ["train", "valid", "test"]
    x = []
    y_truth = []
    for file in files:
        with open(os.path.join(file_path, f"{file}.csv"), "r") as rd:
            for i, line in enumerate(rd.readlines()):
                values = line.strip().split(',')
                x.append((int(values[0]), int(values[1])))
                y_truth.append(int(values[2]))
    return x, y_truth


def rel_serialize(cols: List[str], vals: List[str], skip=True, add_token=True) -> str:
    sen = ""
    for (col, val) in zip(cols, vals):
        if skip and val == "":
            continue
        if col.lower() == "id":
            continue
        if add_token:
            sen += f"COL {col} VAL {val} "
        else:
            sen += f"{col} {val} "
    return sen


def semi_serialize(line: dict, skip=True, add_token=True) -> str:
    sen = ""
    for (key, val) in line.items():
        if key == "id":
            continue
        if type(val).__name__ == "list":
            if skip and len(val) == 0:
                continue
            if add_token:
                sen += f"COL {key} VAL {' '.join(list(map(str, val)))} "
            else:
                sen += f"{key} {' '.join(list(map(str, val)))} "
        elif type(val).__name__ == "dict":
            if add_token:
                sen += f"COL {key} VAL {semi_serialize(val, skip, add_token)} "
            else:
                sen += f"{key} {semi_serialize(val, skip, add_token)} "
        else:
            val = str(val)
            if skip and val == "":
                continue
            if add_token:
                sen += f"COL {key} VAL {val} "
            else:
                sen += f"{key} {val} "

    return sen


def read_rel_entities(file_path: str, add_token=True, summarize=False):
    entities = []
    file_path += ".csv"
    with open(file_path, "r") as rd:
        data = list(csv.reader(rd))
        cols = data[0]
        for vals in tqdm(data[1:], desc="read relation entity..."):
            entities.append(rel_serialize(cols, vals, add_token=add_token))
    if summarize:
        summarizer = Summarizer(entities, "roberta-base")
        new_entities = []
        for ent in tqdm(entities, desc="summarizing..."):
            new_entities.append(summarizer.transform_sentence(ent))
        entities = new_entities
    return entities


def read_semi_entities(file_path: str, skip=True, add_token=True, summarize=False):
    entities = []
    file_path += ".json"
    with open(file_path, "r") as rd:
        lines = json.load(rd)
        for line in tqdm(lines, desc="read semi entity..."):
            entities.append(semi_serialize(line, skip, add_token=add_token))
    if summarize:
        summarizer = Summarizer(entities, "roberta-base")
        new_entities = []
        for ent in tqdm(entities, desc="summarizing..."):
            new_entities.append(summarizer.transform_sentence(ent))
        entities = new_entities
    return entities


def read_text_entities(file_path: str, add_token=False, summarize=False):
    entities = []
    file_path += ".txt"
    with open(file_path, "r") as rd:
        lines = rd.readlines()
        for line in tqdm(lines, desc="read text entity..."):
            text = line.strip()
            entities.append(text)
    if summarize:
        summarizer = Summarizer(entities, "roberta-base")
        new_entities = []
        for ent in tqdm(entities, desc="summarizing..."):
            new_entities.append(summarizer.transform_sentence(ent))
        entities = new_entities
    return entities


read_entities_funs = {
    "rel-heter": (read_rel_entities, read_rel_entities),
    "semi-homo": (read_semi_entities, read_semi_entities),
    "semi-heter": (read_semi_entities, read_semi_entities),
    "semi-rel": (read_rel_entities, read_semi_entities),
    "semi-text-c": (read_semi_entities, read_text_entities),
    "semi-text-w": (read_semi_entities, read_text_entities),
    "rel-text": (read_text_entities, read_rel_entities)
}


def read_entities(data_type: str, args: PromptEMArgs):
    read_entities_fun = read_entities_funs[data_type]
    left_entities = read_entities_fun[0](f"data/{data_type}/left", add_token=args.add_token,
                                         summarize=args.text_summarize)
    right_entities = read_entities_fun[1](f"data/{data_type}/right", add_token=args.add_token,
                                          summarize=args.text_summarize)
    return left_entities, right_entities


def read_ground_truth_few_shot(file_path, files, k=0.1, seed=2022, return_un_y=False):
    set_seed(seed)
    x_pos = []
    x_neg = []
    all_samples = []
    all_samples_y = []
    for file in files:
        with open(os.path.join(file_path, f"{file}.csv"), "r") as rd:
            for i, line in enumerate(rd.readlines()):
                values = line.strip().split(',')
                if int(values[2]) == 1:
                    x_pos.append((int(values[0]), int(values[1])))
                else:
                    x_neg.append((int(values[0]), int(values[1])))
                all_samples.append((int(values[0]), int(values[1])))
                all_samples_y.append(int(values[2]))
    x = []
    if isinstance(k, float):
        num_sample_pos = round(len(x_pos) * k)
        num_sample_neg = round(len(x_neg) * k)
    else:
        num_sample_pos = min(k, len(x_pos))
        num_sample_neg = min(k, len(x_neg))
    logging.info(f"num_sample_pos: {num_sample_pos}")
    logging.info(f"num_sample_neg: {num_sample_neg}")
    x.extend(random.sample(x_pos, num_sample_pos))
    x.extend(random.sample(x_neg, num_sample_neg))
    y_truth = np.concatenate((np.ones(num_sample_pos, dtype=int), np.zeros(num_sample_neg, dtype=int))).tolist()
    for item in x:
        idx = all_samples.index(item)
        all_samples_y.pop(idx)
        all_samples.remove(item)
    if return_un_y:
        return x, y_truth, all_samples, all_samples_y
    else:
        return x, y_truth, all_samples


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_unique_label_trees(root_tree: cluster.hierarchy.ClusterNode, labels, max_dist=None, path='root'):
    if max_dist is None:
        max_dist = np.inf

    if root_tree.is_leaf():
        return [(root_tree, path)]
    found_labels = [labels[ix] for ix in root_tree.pre_order() if labels[ix] != -1]

    # Case when tree contains only unlabelled samples
    if len(found_labels) == 0:
        if root_tree.dist < max_dist:
            return [(root_tree, path)]

    # Case when tree contains at most 1 unique label
    elif len(set(found_labels)) == 1:
        if root_tree.dist < max_dist:
            return [(root_tree, path)]

    # Fallback case: more than 2 unique labels in the tree, or distance objective not reached
    return (get_unique_label_trees(root_tree.left, labels, max_dist, path=f'{path}.left') +
            get_unique_label_trees(root_tree.right, labels, max_dist, path=f'{path}.right'))


def statistic_of_current_train_set(data: PromptEMData):
    acc = 0
    neg = 0
    pos = 0
    for i, pair in enumerate(data.train_pairs):
        t = (pair[0], pair[1])
        if data.train_y[i] == 1:
            pos += 1
        else:
            neg += 1
        if t in data.ground_truth:
            acc += int(1 == data.train_y[i])
        else:
            acc += int(0 == data.train_y[i])
    siz = len(data.train_pairs)
    per = len(data.train_pairs) / (len(data.train_pairs) + len(data.train_un_pairs))
    acc = 0 if siz == 0 else acc / len(data.train_pairs)
    return siz, pos, neg, per, acc


def EL2N_score(p, y):
    """
    :param p: torch.Size([batch_size,2])
    :param y: torch.Size([batch_size])
    :return:
    """
    y = one_hot(y, num_classes=2)
    dis = torch.norm(p - y, p=2, dim=1)
    return dis
