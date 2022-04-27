"""
    Modified by https://github.com/megagonlabs/ditto/blob/master/ditto_light/summarize.py
"""
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords
from transformers import AutoTokenizer

stopwords = set(stopwords.words('english'))


class Summarizer:
    """To summarize a data entry pair into length up to the max sequence length.
    Args:
        task_config (Dictionary): the task configuration
        lm (string): the language model (bert, albert, or distilbert)
    Attributes:
        config (Dictionary): the task configuration
        tokenizer (Tokenizer): a tokenizer from the huggingface library
    """

    def __init__(self, entities, lm):
        self.entities = entities
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.len_cache = {}

        # build the tfidf index
        self.build_index()

    def build_index(self):
        """Build the idf index.
        Store the index and vocabulary in self.idf and self.vocab.
        """
        content = []
        for line in self.entities:
            content.append(line)

        vectorizer = TfidfVectorizer().fit(content)
        self.vocab = vectorizer.vocabulary_
        self.idf = vectorizer.idf_

    def get_len(self, word):
        """Return the sentence_piece length of a token.
        """
        if word in self.len_cache:
            return self.len_cache[word]
        length = len(self.tokenizer.tokenize(word))
        self.len_cache[word] = length
        return length

    def transform_sentence(self, sent, max_len=512):
        res = ''
        cnt = Counter()
        tokens = sent.split(' ')
        for token in tokens:
            if token not in ['COL', 'VAL'] and \
                    token not in stopwords:
                if token in self.vocab:
                    cnt[token] += self.idf[self.vocab[token]]

        token_cnt = Counter(sent.split(' '))
        total_len = token_cnt['COL'] + token_cnt['VAL']

        subset = Counter()
        # for token in set(token_cnt.keys()):
        data = list(set(token_cnt.keys()))
        data.sort()
        random.Random(2022).shuffle(data)
        for token in data:
            subset[token] = cnt[token]
        # ZXC Fix https://docs.python.org/3/library/collections.html#collections.Counter.most_common
        # most_common对于相同频率的word是按出现(插入)顺序进行排序，所以问题出在上面对set的遍历顺序不确定
        subset = subset.most_common(max_len)

        topk_tokens_copy = set([])

        for word, _ in subset:
            bert_len = self.get_len(word)
            if total_len + bert_len > max_len:
                break
            total_len += bert_len
            topk_tokens_copy.add(word)
        for token in sent.split(' '):
            if token in ['COL', 'VAL']:
                res += token + ' '
            elif token in topk_tokens_copy:
                res += token + ' '
                topk_tokens_copy.remove(token)
        return res
