import os
import re
import numpy as np
from tqdm import tqdm
from imdb import ImdbDataset
from torch.utils.data import DataLoader, Dataset
import evaluate

r""" Base on 知网dict

基本逻辑:

    1. 首先需要找出句子里面的情感词，然后再找出程度副词和否定词等。
        得出这句话的一个正面分值，一个负面分值(消极分值也是正数， 无需使用负数),
        以分句的情感为基础，加权求和，从而得到一条评论的情感分值
    
    2. 发现叹号可以为情感值+2.  
    
1. 程度副词 权重分配
most，very，more，ish，insufficient  over
 4，   3，   2，    0.5，-0.3        -0.5

"""


class CNKISentDict:
    
    def __init__(self, dict_path):
        self.path = dict_path
        
    def predict(self, X):
        r"""predict 

        Args:
            X (list or array): 
        """
        adv_of_degree, pos_words, neg_words = self.read_CNKI(self.path)
        predict_prob = []   # 记录正向得分和负向得分

        adv_degree = ['8','6','4','3','-2','-4']
        for idx, sentences in enumerate(X):
            # 1. 分句
            predict_prob.append([0,0])
            sentences = re.sub('\.|,','<break>',sentences)
            each_sentence = sentences.split('<break>') # ['sent1','sent2','sent3]
            # print(each_sentence)
            
            temp_prob = np.zeros((len(each_sentence),2))
            for idy,sent in enumerate(each_sentence):
                # 2. 对每一个分句找情感词
                sent = sent.lower()
                sent = sent.strip()
                # print(f"sent:\t{sent}")
                FIND_POS = 0
                FIND_NEG = 0
                for pos in pos_words:
                    FLAG = 0
                    for pos_each in pos.split():
                        if pos_each.strip() in sent:
                            FLAG +=1
                    if FLAG == len(pos.split()):
                        FIND_POS += 1
                        # print("FIND_POS!",pos)
                temp_prob[idy][0] = FIND_POS 
                     
                for neg in neg_words:    
                    FLAG = 0
                    for neg_each in neg.split():
                        if neg_each.strip() in sent:
                            FLAG +=1
                    if FLAG == len(neg.split()):
                        FIND_NEG += 1
                        # print("FIND_NEG!",neg)
                temp_prob[idy][1] = FIND_NEG   
                                       
                for degree in adv_degree:
                    for adv in adv_of_degree[str(degree)]:
                        if adv in sent:
                            temp_prob[idy][0] *= float(degree)
                            temp_prob[idy][1] *= float(degree)
                            # print("FIND_ADV!",adv)
                        if '!' in sent and FIND_POS!=0:
                            temp_prob[idy][0] += 2
                        if '!' in sent and FIND_NEG!=0:
                            temp_prob[idy][1] += 2
            # print(temp_prob)
            predict_prob[idx][0] = np.sum(temp_prob,axis = 0)[0]
            predict_prob[idx][1] = np.sum(temp_prob,axis = 0)[1]
            predict_label = np.argmax(predict_prob,axis = 1)
            
        return  predict_prob, predict_label
    
    @staticmethod
    def read_CNKI(path):
        
        adv_of_degree_path = os.path.join(path,'程度级别词语（英文）.txt')
        neg_comments_path = os.path.join(path, '负面评价词语（英文）.txt')
        neg_sentiments_path = os.path.join(path,'负面情感词语（英文）.txt')
        pos_comments_path = os.path.join(path, '正面评价词语（英文）.txt')
        pos_sentiments_path = os.path.join(path, '正面情感词语（英文）.txt')
    
        # Create adv of degree .JSON file
        with open(adv_of_degree_path,'r') as f:
            adv_of_degree = f.readlines()
        
        most_length = 0
        very_length = 0
        more_length = 0
        ish_length = 0
        insufficiently_length = 0
        over_length = 0
        
        
        adv_of_degree_json = {}
        for idx, text in enumerate(adv_of_degree):
            if re.findall('(极其).*?(\d.)',text):
                most_length = re.findall('(极其).*?(\d.)',text)[0][1]
                adv_of_degree_json['8'] = []
                continue
            if re.findall('(很).*?(\d.)',text):
                very_length = re.findall('(很).*?(\d.)',text)[0][1]
                adv_of_degree_json['6'] = []
                continue
            if re.findall('(较).*?(\d.)',text):
                more_length = re.findall('(较).*?(\d.)',text)[0][1]
                adv_of_degree_json['4'] = []
                continue
            if re.findall('(稍).*?(\d.)',text):
                ish_length = re.findall('(稍).*?(\d.)',text)[0][1]
                adv_of_degree_json['3'] = []
                continue
            if re.findall('(欠).*?(\d.)',text):
                insufficiently_length = re.findall('(欠).*?(\d.)',text)[0][1]
                adv_of_degree_json['-2'] = []
                continue
            if re.findall('(超).*?(\d.)',text):
                over_length = re.findall('(超).*?(\d.)',text)[0][1]
                adv_of_degree_json['-4'] = []
                continue
            
            if most_length != 0 and very_length == 0 and re.sub('\\t|\\n','',text) != '':
                    adv_of_degree_json['8'].append(re.sub('\\t|\\n','',text).strip())
            if very_length != 0 and more_length == 0 and re.sub('\\t|\\n','',text) != '':
                adv_of_degree_json['6'].append(re.sub('\\t|\\n','',text).strip()) 
            if more_length !=0 and ish_length == 0 and re.sub('\\t|\\n','',text) != '':
                adv_of_degree_json['4'].append(re.sub('\\t|\\n','',text).strip())
            if ish_length !=0 and insufficiently_length ==0 and re.sub('\\t|\\n','',text) != '':
                adv_of_degree_json['3'].append(re.sub('\\t|\\n','',text).strip())    
            if insufficiently_length!=0 and over_length ==0 and re.sub('\\t|\\n','',text) != '':
                adv_of_degree_json['-2'].append(re.sub('\\t|\\n','',text).strip())   
            if over_length !=0 and re.sub('\\t|\\n','',text) != '':
                adv_of_degree_json['-4'].append(re.sub('\\t|\\n','',text).strip())
                
            # with open("./cnki dict/en_json/adv_of_degree.json" ,'w' ) as f:
                # json.dump(adv_of_degree_json, f)
        
        with open(neg_comments_path, 'r') as f:
            neg_comments_words = f.readlines()
        for idx, text in enumerate(neg_comments_words):
            if '...' in text:
                text = re.sub('...', '',text)
            if 'be' in text.split():
                text = re.sub('be ', '',text)
                # neg_comments_words[idx] = text.strip()
                # print(neg_comments_words[idx])
            neg_comments_words[idx] = text.strip()
            
        with open(neg_sentiments_path, 'r') as f:
            neg_sentiments_words = f.readlines()   
        for idx, text in enumerate(neg_sentiments_words):
            if '...' in text:
                text = re.sub('...', '',text)
            if 'be' in text.split():
                text = re.sub('be ', '',text)
            neg_sentiments_words[idx] = text.strip()
                
        with open(pos_comments_path, 'r') as f:
            pos_comments_words = f.readlines()
        for idx, text in enumerate(pos_comments_words):
            if '...' in text:
                text = re.sub('...', '',text)
            if 'be' in text.split():
                text = re.sub('be ', '',text)
            pos_comments_words[idx] = text.strip()
                
        with open(pos_sentiments_path, 'r') as f:
            pos_sentiments_words = f.readlines()
        for idx, text in enumerate(pos_sentiments_words):
            if '...' in text:
                text = re.sub('...', '',text)
            if 'be' in text.split():
                text = re.sub('be ', '',text)
            pos_sentiments_words[idx] = text.strip()
            
        pos_words = pos_comments_words + pos_sentiments_words
        neg_words = neg_comments_words + neg_sentiments_words
        pos_words = list(set(pos_words))
        neg_words = list(set(neg_words))
        # print(pos_words)
        return adv_of_degree_json, pos_words, neg_words
        
if __name__ == "__main__":
    dataset = ImdbDataset()
    model = CNKISentDict('./cnki dict/en')
    
    BATCH_SIZE = 64
    dataloader = DataLoader(dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = True)
    
    acc = 0
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    with tqdm(desc="predict", total=len(dataloader), colour="red") as t:
        for i, (text, label) in enumerate(dataloader):
            _, out = model.predict(text)
            t.update(1)
            
            # for j in range(len(out)):
            #     if out[j] == label[j]:
            #         acc+=1
            acc.add_batch(predictions = out,
                         references = label)
            f1.add_batch(predictions = out,
                         references = label)
    print('Accuracy:{:.2f}%,F1:{:.2f} \n'.format(acc.compute()['accuracy']*100., f1.compute()['f1']))   
    
    # Accuracy:49.52%,F1:0.48 