import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict, Counter

# torch 类
import torch
from torch.utils.data import Dataset, TensorDataset
from torchtext.transforms import VocabTransform
from torchtext.vocab import vocab


class BuildVocab(Dataset):
    r"""
    将imdb数据集转化为可读的训练模型
    
    examples:
    >>> imdb_dataset = BuildVocab()
    >>> imdb_dataset.build_vocab(dataset = ImdbDataset(),min_count=1)
    >>> imdb_dataset.make_dataset(dataset = ImdbDataset(split='train'),max_length=100)
    """
    def __init__(self):
        super(BuildVocab, self).__init__()
        self.dataset = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_row = self.dataset[idx][0]
        data_label = self.dataset[idx][1] 
        return data_row, data_label
        
    def build_vocab(self, dataset:Dataset,min_count=2):
        """ 
        from imdb Dataset to build a vocabulary
        """
        # 读入每一句，就对其进行拆分，然后使用Counter进行计数，构建计数表，通过迭代数据集，更新counter
        
        with tqdm(desc="make vocab", total=len(dataset), colour="red") as t:
            for i, x in enumerate(dataset):
                text = x[0]
                word_list = text.split()
                # word_list = x[0]
                t.update(1)
                if i == 0:
                    counter = Counter(word_list)
                else:
                    counter.update(word_list)
        
        # 将计数表按照 value的大小进行倒序排序
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 转化为 OrderedDict 
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        # 加入特殊字符，包括未知字符 和 填充字符
        special_tokens = ['<UNK>', '<PAD>']
        # 构建词汇表
        # {(0, '<UNK>'),(1, '<PAD>'),(2, 'the'),...} 
        # 2 以后按照出现的频率从高到低进行排序
        text_vocab = vocab(ordered_dict,min_freq= min_count, specials=special_tokens)
        
        # 设置默认下标，当出现了新词汇的时候(不在已经构建的词汇表中，也就是不认识这个词)，将其对应对应的下标为 default_index
        # 一般default_index也就是'<UNK>'，也就是设置两个下标相同.
        text_vocab.set_default_index(0)
        # vocab_transform 可以将词汇转化为下标
        # >>> vocab_transform(['<UNK>','<PAD>'])
        # [0, 1]
        
        vocab_transform = VocabTransform(text_vocab)
        # print(type(text_vocab))
        # print(type(vocab_transform))
        self.text_vocab = text_vocab
        self.vocab_transform = vocab_transform
        self.vocab_size = len(self.text_vocab)
        
    def make_dataset(self,dataset:Dataset, save_path:Optional[str] = None, max_length = 200):
        data_list = []
        label_list = []
        with tqdm(desc='make dataset',total=len(dataset),colour='green') as t:
            idx = 0
            for x in dataset:
                # print(x)
                # break
                label = x[1]
                text = x[0]
                sentence_words = text.split(' ')  # 切分句子
                sentence_id_list = np.array(self.vocab_transform(sentence_words))
                FLAG, sentence_id_list = self.pad_or_cut(sentence_id_list, max_length)
                
                # NOTE 
                if FLAG:
                    for i in range(len(sentence_id_list)):
                        data_list.append([])
                        data_list[idx] = sentence_id_list[i]
                        label_list.append(label)
                        idx += 1
                        
                else:
                    data_list.append([])
                    data_list[idx] = sentence_id_list
                    label_list.append(label)
                    idx += 1
                t.update(1)
        # NOTE: list 先转化为 ndarray 再转化为 tensor 速度比 从list直接转化为 tensor 快
        
        data_list = np.array(data_list)
        label_list = np.array(label_list)
        
        data_list = torch.tensor(data_list)
        label_list = torch.tensor(label_list)
        train_dataset = TensorDataset(data_list,label_list)
        
        # 保存模型
        if save_path:
            torch.save(train_dataset,save_path)
        
        self.dataset = train_dataset
    
    def pad_or_cut(self,value: np.ndarray, max_length: int):
        """填充或截断一维numpy到固定的长度"""
        RESHAPE = False
        if len(value) < max_length:  # 填充
            data_row = np.concatenate((value,self.vocab_transform(['<PAD>'])*(max_length-len(value))))
            
        else:  # 截断
            idx = 1
            while True:
                if max_length * idx >len(value):
                    pad = max_length * idx - len(value)
                    break
                else:
                    idx += 1
            data_row = np.concatenate((value,self.vocab_transform(['<PAD>'])*pad))
            data_row = data_row.reshape((-1, max_length))
            RESHAPE = True
        return RESHAPE, data_row
