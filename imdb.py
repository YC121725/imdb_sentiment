import os
import re
from torch.utils.data import Dataset
from typing import Optional
import datasets

class ImdbDataset(Dataset):
    def __init__(self, split:Optional[str] = None):
        
        self.total_path = []  # 保存所有的文件路径
        
        if split is not None:
            if split == 'train':
                data_path = "./data/train" 
            if split == 'test':
                data_path = "./data/test"

            for temp_path in ["pos", "neg"]:
                cur_path = os.path.join(data_path,temp_path)
                self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]
        
        if split == None:
            data_path = ["./data/train", "./data/test"]
        
            for temp_path in ["pos", "neg"]:
                for each_data_path in data_path:
                    cur_path = os.path.join(each_data_path,temp_path)

                    self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")] 
        
    def __getitem__(self, idx):
        file = self.total_path[idx]
        
        # 从txt获取评论并分词
        review = self.tokenlize(open(file, "r", encoding="utf-8").read())
        # # 获取评论对应的label
        label = int(file.split("_")[-1].split(".")[0])
        label = 0 if label < 5 else 1
        
        return review, label

    def __len__(self):
        return len(self.total_path)
    
    @staticmethod
    def tokenlize(sentence):
        """
        进行文本分词
        :param sentence: str
        :return: [str,str,str]
        """
        filters = [
                "!",'"',"#","$","%","&","\(","\)","\*","\+",",",
                "-","\.","/",":",";","<","=",">","\?","@","\[","\\","\]",
                "^","_","`","\{","\|","\}","~","\t","\n","\x85","\x806","\x91","\x97","\x96","”","“",
            ]
        sentence = sentence.lower()  # 把大写转化为小写
        sentence = re.sub("<.*?>", " ", sentence, flags=re.S)
        sentence = re.sub("|".join(filters), " ", sentence)
        sentence = re.sub("^[0-9]*$"," ",sentence)
        # result = [i for i in sentence.split(" ") if len(i) > 0]

        return sentence
    
    @staticmethod
    def from_dataset():
        r""" imdb dataset
        
        Convert `torch.utils.data.Dataset` to `datasets.Dataset`
        
        Args:
            train: training dataset
            test: testing dataset  
        """
        
        train_dataset = ImdbDataset(split='train')
        test_dataset = ImdbDataset(split='test')
        
        train_text = [i[0] for i in train_dataset]
        train_label = [i[1] for i in train_dataset]

        text_text = [i[0] for i in test_dataset]
        text_label = [i[1] for i in test_dataset]

        traindata = datasets.Dataset.from_dict({'text':train_text,
                                                'label':train_label})
        testdata = datasets.Dataset.from_dict({'text':text_text,
                                            'label':text_label})

        hfdataset = datasets.DatasetDict({'train':traindata,
                                        'test':testdata})
        return hfdataset