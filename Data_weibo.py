import spacy

from collections import Counter
from operator import itemgetter
from collections import OrderedDict
from random import choice
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn.functional as fn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import codecs
def get_emotion_dict():
    lines = None
    with codecs.open('emotion_sentiment.txt', mode='r', encoding='utf8') as f:
        lines = f.readlines()

    lines = [x.strip().split('\t') for x in lines]
    emoji_dict = {x[0]:x[1] for x in lines}
    return emoji_dict

class TuplesListDataset(Dataset):
    def __init__(self, tuplelist):
        super(TuplesListDataset, self).__init__()
        self.tuplelist = tuplelist
        self.mappings = {}

    def __len__(self):
        return len(self.tuplelist)

    def __getitem__(self,index):
        if len(self.mappings) == 0:
            return self.tuplelist[index]
        else:
            t = list(self.tuplelist[index])

            for i,m in self.mappings.items():
                if t[i] in m:
                    t[i] = m[t[i]]
                else:
                    t[i] = 0

            return tuple(t)

    def __iter__(self):
        return self.tuplelist.__iter__()


    def field_gen(self,field,transform=False):
        if transform:
            for i in range(len(self)):
                yield self[i][field]
        else:
            for x in self:
                yield x[field]


    def get_stats(self,field):
        d =  dict(Counter(self.field_gen(field)))
        sumv = sum([v for k,v in d.items()])
        class_per = {k:(v/sumv) for k,v  in d.items()}

        return d,class_per

    def get_field_dict(self, field,  offset=0,  dict=None):

       if dict is not None:
            d2k={}
            for i, c in enumerate(set(self.field_gen(field)),offset):
                is_exit = False
                for k, v in dict.items():
                    if c==k:
                        is_exit = True
                        d2k[c] = v
                if not is_exit:
                    d2k[c] = 0
            return d2k

       else:
           d2k = {c: i for i, c in enumerate(set(self.field_gen(field)), offset)}
           return d2k


    def set_mapping(self,field, offset=0, dict=None,  mapping=None,unk=None):

        """
        Sets or creates a mapping for a tuple field. Mappings are {k:v} with keys starting at offset.
        """
        mapping = self.get_field_dict(field, offset, dict )
        self.mappings[field] = mapping

        return mapping


    @staticmethod
    def build_train_test(datatuples,splits,no_testset=False):

        all_train, all_test = [], []
        if no_testset is False:
            total_splits = max(splits)
        else:
            total_splits = 0
        for split_num in range(total_splits+1):
            train, test = [], []
            for split,data in tqdm(zip(splits,datatuples),total=len(datatuples),desc="Building train/test of split #{}".format(split_num)):
                if split == split_num and no_testset == False:
                    test.append(data)
                else:
                    train.append(data)

            all_train.append(train)
            all_test.append(test)

        return [TuplesListDataset(train) for train in all_train], [TuplesListDataset(test) for test in all_test]

class BucketSampler(Sampler):
    """
    Evenly sample from bucket for datalen
    """

    def __init__(self, dataset,field):
        self.dataset = dataset
        self.field = field
        self.index_buckets = self._build_index_buckets()
        self.len = min([len(x) for x in self.index_buckets.values()])

    def __iter__(self):
        return iter(self.bucket_iterator())

    def __len__(self):
        return self.len

    def bucket_iterator(self):
        cl = list(self.index_buckets.keys())
   
        for x in range(len(self)):
            yield choice(self.index_buckets[choice(cl)])

            
    def _build_index_buckets(self):
        class_index = {}
        for ind,cl in enumerate(self.dataset.field_gen(self.field,True)):
            if cl not in class_index:
                class_index[cl] = [ind]
            else:
                class_index[cl].append(ind)
        return class_index

class Vectorizer():

    def __init__(self,word_dict=None,max_sent_len=8,max_word_len=32):
        self.word_dict = word_dict
        self.nlp = spacy.load('en')
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len

    def _get_words_dict(self,data,max_words):
        word_counter = Counter(w.lower_ for d in self.nlp.tokenizer.pipe((doc for doc in tqdm(data,desc="Tokenizing data"))) for w in d)
        dict_w =  {w: i for i,(w,_) in tqdm(enumerate(word_counter.most_common(max_words),start=2),desc="building word dict",total=max_words)}
        dict_w["_padding_"] = 0
        dict_w["_unk_word_"] = 1
        print("Dictionnary has {} words".format(len(dict_w)))
        return dict_w

    def build_dict(self,text_iterator,max_f):
        self.word_dict = self._get_words_dict(text_iterator,max_f)

    def vectorize_batch(self,t,trim=True,is_emoji=False):
        return self._vect_dict(t,trim,is_emoji)

    def _vect_dict(self,t,trim,is_emoji=False):

        if self.word_dict is None:
            print("No dictionnary to vectorize text \n-> call method build_dict \n-> or set a word_dict attribute \n first")
            raise Exception

        if is_emoji:
            emoji_label = []
            for k, word in enumerate(t):
                if word in self.word_dict:
                    emoji_label.append(self.word_dict[word])
                else:
                    emoji_label.append(self.word_dict["_unk_word_"])  # _unk_word_
            return torch.LongTensor(emoji_label)

        revs = []
        for rev in t:
            review = []
            rev = rev.strip()
            words = rev.split(' ')
            s = []
            for k, word in enumerate(words):
                if word in self.word_dict:
                    s.append(self.word_dict[word])
                else:
                    s.append(self.word_dict["_unk_word_"])  # _unk_word_

            review.append(torch.LongTensor(s))
            revs.append(review)

        return revs

emotion_dict = get_emotion_dict()

def get_inverse_dict(dict):
    return  {v: k for k, v in dict.items()}