import logging
import random
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LMNet
from pseudo_label import *
from args import PromptEMArgs
from data import PromptEMData, TypeDataset
from prompt import get_prompt_model, get_prompt_dataloader, read_prompt_dataset
from utils import evaluate, statistic_of_current_train_set, EL2N_score


def train_plm(args: PromptEMArgs, model, labeled_train_dataloader, optimizer, scaler):
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_total = []
    for batch in tqdm(labeled_train_dataloader):
        x, labels = batch
        x = torch.tensor(x).to(args.device)
        labels = torch.tensor(labels).to(args.device)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_total.append(loss.item())
    return np.array(loss_total).mean()


def eval_plm(args: PromptEMArgs, model, data_loader, return_acc=False):
    model.eval()
    y_truth = []
    y_pre = []
    for batch in tqdm(data_loader):
        x, labels = batch
        x = torch.tensor(x).to(args.device)
        y_truth.extend(labels)
        with torch.no_grad():
            logits = model(x)
            logits = torch.argmax(logits, dim=1)
            logits = logits.cpu().numpy().tolist()
            y_pre.extend(logits)
    return evaluate(np.array(y_truth), np.array(y_pre), return_acc=return_acc)


def train_prompt(args: PromptEMArgs, model, labeled_train_dataloader, optimizer, scaler):
    model.train()
    loss_fn = CrossEntropyLoss()
    loss_total = []
    for _batch in tqdm(labeled_train_dataloader):
        labeled_batch = copy.deepcopy(_batch)
        labeled_batch = labeled_batch.to(args.device)
        y_truth = labeled_batch.label
        with autocast():
            logits = model(labeled_batch)
            loss = loss_fn(logits, y_truth)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_total.append(loss.item())
    return np.array(loss_total).mean()


def eval_prompt(args: PromptEMArgs, model, data_loader, return_acc=False):
    model.eval()
    y_truth_all = []
    y_pred_all = []
    for batch in data_loader:
        batch = batch.to(args.device)
        y_truth = batch.label
        with torch.no_grad():
            logits = model(batch)
            y_pred = torch.argmax(logits, dim=-1)
            y_truth_all.extend(y_truth.cpu().numpy().tolist())
            y_pred_all.extend(y_pred.cpu().numpy().tolist())
    return evaluate(np.array(y_truth_all), np.array(y_pred_all), return_acc=return_acc)


def pruning_dataset(args: PromptEMArgs, data: PromptEMData, model, prompt=True) -> int:
    if prompt:
        labeled_dataset = read_prompt_dataset(data.left_entities, data.right_entities, data.train_pairs, data.train_y)
        labeled_dataloader = get_prompt_dataloader(args, labeled_dataset, shuffle=False)
    else:
        labeled_dataset = TypeDataset(data, "train")
        labeled_dataloader = DataLoader(dataset=labeled_dataset, batch_size=args.batch_size, collate_fn=TypeDataset.pad)
    model.eval()
    all_pos_el2n = []
    all_neg_el2n = []
    for batch in tqdm(labeled_dataloader, desc="pruning..."):
        if hasattr(batch, "to"):
            batch = batch.to(args.device)
            y_truth = batch.label
        else:
            x, labels = batch
            x = torch.tensor(x).to(args.device)
            labels = torch.tensor(labels).to(args.device)
        with torch.no_grad():
            out_prob = []
            # mc-el2n
            for _ in range(args.mc_dropout_pass):
                if hasattr(batch, "to"):
                    _batch = copy.deepcopy(batch)
                    logits = model(_batch)
                else:
                    logits = model(x)
                logits = torch.softmax(logits, dim=-1)
                out_prob.append(logits.detach())
            out_prob = torch.stack(out_prob)
            out_prob = torch.mean(out_prob, dim=0)
            out_prob = out_prob.detach()
            if hasattr(batch, "to"):
                y_truth = y_truth.detach()
            else:
                y_truth = labels
            pos_el2n = EL2N_score(out_prob[y_truth == 1], y_truth[y_truth == 1])
            neg_el2n = EL2N_score(out_prob[y_truth == 0], y_truth[y_truth == 0])
            all_pos_el2n.extend(pos_el2n)
            all_neg_el2n.extend(neg_el2n)
    k = int(args.el2n_ratio * len(all_pos_el2n))
    values, indices = torch.topk(torch.tensor(all_pos_el2n), k=k)
    pos_ids = indices.numpy().tolist()
    k = int(args.el2n_ratio * len(all_neg_el2n))
    values, indices = torch.topk(torch.tensor(all_neg_el2n), k=k)
    neg_ids = indices.numpy().tolist()
    ids = pos_ids + neg_ids
    data.train_pairs = [x for (i, x) in enumerate(data.train_pairs) if i not in ids]
    data.train_y = [x for (i, x) in enumerate(data.train_y) if i not in ids]
    return len(ids)


class BestMetric:
    def __init__(self):
        self.valid_f1 = -1
        self.test_metric = None
        self.state_dict = None


def inner_train(args: PromptEMArgs, model, optimizer, scaler, train_dataloader, valid_dataloader, test_dataloader,
                prompt=True):
    if prompt:
        loss = train_prompt(args, model, train_dataloader, optimizer, scaler)
    else:
        loss = train_plm(args, model, train_dataloader, optimizer, scaler)
    logging.info(f"loss: {loss}")
    if prompt:
        valid_p, valid_r, valid_f1 = eval_prompt(args, model, valid_dataloader)
    else:
        valid_p, valid_r, valid_f1 = eval_plm(args, model, valid_dataloader)
    logging.info(f"[Valid] Precision: {valid_p:.4f}, Recall: {valid_r:.4f}, F1: {valid_f1:.4f}")
    if prompt:
        test_p, test_r, test_f1 = eval_prompt(args, model, test_dataloader)
    else:
        test_p, test_r, test_f1 = eval_plm(args, model, test_dataloader)
    logging.info(f"[Test] Precision: {test_p:.4f}, Recall: {test_r:.4f}, F1: {test_f1:.4f}")
    return (valid_p, valid_r, valid_f1, test_p, test_r, test_f1)


def update_best(model, metric, best: BestMetric):
    valid_p, valid_r, valid_f1, test_p, test_r, test_f1 = metric
    if valid_f1 > best.valid_f1 or valid_f1 == best.valid_f1 and test_f1 > best.test_metric[2]:
        best.valid_f1 = valid_f1
        best.test_metric = (test_p, test_r, test_f1)
        best.state_dict = model.state_dict()


def train_and_update_best(args: PromptEMArgs, model, optimizer, scaler, train_dataloader, valid_dataloader,
                          test_dataloader, best: BestMetric, prompt=True):
    metric = inner_train(args, model, optimizer, scaler, train_dataloader, valid_dataloader, test_dataloader,
                             prompt)
    update_best(model, metric, best)


def self_training(args: PromptEMArgs, data: PromptEMData):
    train_set = read_prompt_dataset(data.left_entities, data.right_entities, data.train_pairs, data.train_y)
    valid_set = read_prompt_dataset(data.left_entities, data.right_entities, data.valid_pairs, data.valid_y)
    test_set = read_prompt_dataset(data.left_entities, data.right_entities, data.test_pairs, data.test_y)
    valid_loader = get_prompt_dataloader(args, valid_set, shuffle=False)
    test_loader = get_prompt_dataloader(args, test_set, shuffle=False)
    train_loader = get_prompt_dataloader(args, train_set, shuffle=True)
    best = BestMetric()
    for iter in range(1, args.num_iter + 1):
        # train the teacher model
        model, tokenizer, wrapperClass, template = get_prompt_model(args)
        model.to(args.device)
        optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
        scaler = GradScaler()
        siz, pos, neg, per, acc = statistic_of_current_train_set(data)
        logging.info(f"[Current Train Set] Size: {siz} Pos: {pos} Neg: {neg} Per: {per:.2f} Acc: {acc:.4f}")
        for epoch in range(1, args.teacher_epochs + 1):
            logging.info(f"[Teacher] epoch#{epoch}")
            train_and_update_best(args, model, optimizer, scaler, train_loader, valid_loader, test_loader, best)
        p, r, f1 = best.test_metric
        logging.info(f"[Best Teacher in iter#{iter}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        if args.self_training:
            # generate pseudo labels
            num = gen_pseudo_labels(args, data, model)
            logging.info(f"[Add Pseudo Labels] Size: {num}")
            siz, pos, neg, per, acc = statistic_of_current_train_set(data)
            logging.info(f"[Current Train Set] Size: {siz} Pos: {pos} Neg: {neg} Per: {per:.2f} Acc: {acc:.4f}")
            # update train dataloader
            train_set = read_prompt_dataset(data.left_entities, data.right_entities, data.train_pairs, data.train_y)
            train_loader = get_prompt_dataloader(args, train_set, shuffle=True)
            model, tokenizer, wrapperClass, template = get_prompt_model(args)
            model.to(args.device)
            optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
            scaler = GradScaler()
            for epoch in range(1, args.student_epochs + 1):
                logging.info(f"[Student] epoch#{epoch}")
                train_and_update_best(args, model, optimizer, scaler, train_loader, valid_loader, test_loader,
                                      best)
                # dynamic dataset
                if args.dynamic_dataset != -1 and (epoch % args.dynamic_dataset) == 0 and epoch != args.student_epochs:
                    num = pruning_dataset(args, data, model)
                    logging.info(f"[Remove Pairs] Size: {num}")
                    siz, pos, neg, per, acc = statistic_of_current_train_set(data)
                    logging.info(f"[Current Train Set] Size: {siz} Pos: {pos} Neg: {neg} Per: {per:.2f} Acc: {acc:.4f}")
                    # update train dataloader
                    train_set = read_prompt_dataset(data.left_entities, data.right_entities, data.train_pairs,
                                                    data.train_y)
                    train_loader = get_prompt_dataloader(args, train_set, shuffle=True)
        p, r, f1 = best.test_metric
        logging.info(f"[Best in iter#{iter}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")


def self_training_only_plm(args: PromptEMArgs, data: PromptEMData):
    train_set = TypeDataset(data, "train")
    valid_set = TypeDataset(data, "valid")
    test_set = TypeDataset(data, "test")
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=TypeDataset.pad, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, collate_fn=TypeDataset.pad)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, collate_fn=TypeDataset.pad)
    best = BestMetric()
    for iter in range(1, args.num_iter + 1):
        # train the teacher model
        model = LMNet()
        model.to(args.device)
        optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
        scaler = GradScaler()
        siz, pos, neg, per, acc = statistic_of_current_train_set(data)
        logging.info(f"[Current Train Set] Size: {siz} Pos: {pos} Neg: {neg} Per: {per:.2f} Acc: {acc:.4f}")
        for epoch in range(1, args.teacher_epochs + 1):
            logging.info(f"[Teacher] epoch#{epoch}")
            train_and_update_best(args, model, optimizer, scaler, train_loader, valid_loader, test_loader, best,
                                  prompt=False)
        p, r, f1 = best.test_metric
        logging.info(f"[Best Teacher in iter#{iter}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        if args.self_training:
            # generate pseudo labels
            num = gen_pseudo_labels(args, data, model, prompt=False)
            logging.info(f"[Add Pseudo Labels] Size: {num}")
            siz, pos, neg, per, acc = statistic_of_current_train_set(data)
            logging.info(f"[Current Train Set] Size: {siz} Pos: {pos} Neg: {neg} Per: {per:.2f} Acc: {acc:.4f}")
            # update train dataloader
            train_set = TypeDataset(data, "train")
            train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=TypeDataset.pad,
                                      shuffle=True)
            # train the student model (freeze the encoder and fine-tune the lstm)
            model = LMNet()
            model.to(args.device)
            optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
            scaler = GradScaler()
            for epoch in range(1, args.student_epochs + 1):
                logging.info(f"[Student] epoch#{epoch}")
                train_and_update_best(args, model, optimizer, scaler, train_loader, valid_loader, test_loader,
                                      best, prompt=False)
                # dynamic dataset
                if args.dynamic_dataset != -1 and (epoch % args.dynamic_dataset) == 0 and epoch != args.student_epochs:
                    num = pruning_dataset(args, data, model, prompt=False)
                    logging.info(f"[Remove Pairs] Size: {num}")
                    siz, pos, neg, per, acc = statistic_of_current_train_set(data)
                    logging.info(f"[Current Train Set] Size: {siz} Pos: {pos} Neg: {neg} Per: {per:.2f} Acc: {acc:.4f}")
                    # update train dataloader
                    train_set = TypeDataset(data, "train")
                    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=TypeDataset.pad,
                                              shuffle=True)
        p, r, f1 = best.test_metric
        logging.info(f"[Best in iter#{iter}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
