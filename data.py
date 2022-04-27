import os
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Vocab:
    def __init__(self, labels_list):
        uniq_labels_list = sorted(set(labels_list))
        self._str_to_ix = {l: ix for ix, l in enumerate(uniq_labels_list)}
        self._ix_to_str = {ix: l for ix, l in enumerate(uniq_labels_list)}
        self.labels = uniq_labels_list

    def __call__(self, x, rev=False):
        if rev:
            return self._ix_to_str[x]
        else:
            return self._str_to_ix[x]


class PromptEMData:
    def __init__(self, data_type) -> None:
        self.data_type = data_type
        self.left_entities = []
        self.right_entities = []
        self.train_pairs = []
        self.train_y = []
        self.train_un_pairs = []
        # only used in test_pseudo_labels, will not be updated
        self.train_un_y = []
        self.valid_pairs = []
        self.valid_y = []
        self.test_pairs = []
        self.test_y = []
        self.ground_truth = set()

    def read_all_ground_truth(self, file_path):
        self.ground_truth = []
        for file in ["train", "valid", "test"]:
            with open(os.path.join(file_path, f"{file}.csv"), "r") as rd:
                for i, line in enumerate(rd.readlines()):
                    values = line.strip().split(',')
                    if int(values[2]) == 1:
                        self.ground_truth.append((int(values[0]), int(values[1])))
        self.ground_truth = set(self.ground_truth)


class TypeDataset(Dataset):
    def __init__(self, data: PromptEMData, mode, lm='roberta-base'):
        self.data = data
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.lm = lm
        self.sentences = []
        self.labels = []
        self.type_vocab = {}
        self.init()
        self.len = len(self.sentences)

    def init(self):
        if self.mode == "train":
            pairs = self.data.train_pairs
            y = self.data.train_y
        elif self.mode == "valid":
            pairs = self.data.valid_pairs
            y = self.data.valid_y
        elif self.mode == "test":
            pairs = self.data.test_pairs
            y = self.data.test_y
        else:
            pairs = self.data.train_un_pairs
            y = []
        for i, pair in enumerate(pairs):
            left = self.data.left_entities[pair[0]]
            right = self.data.right_entities[pair[1]]
            sentence = self.tokenizer(left, right, truncation=True, max_length=512)
            self.sentences.append(sentence['input_ids'])
            if len(y) > 0:
                self.labels.append(y[i])
            else:
                self.labels.append(-1)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        length = len(sentence)
        return sentence, label, length

    @staticmethod
    def pad(batch):
        """
        Pads to the longest sample.
        """
        f = lambda x: [sample[x] for sample in batch]

        lengths = f(2)
        max_len = np.array(lengths).max()

        sentences = f(0)
        labels = f(1)
        for sentence in sentences:
            sentence += [0] * (max_len - len(sentence))
        return sentences, labels
