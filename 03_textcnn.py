from vocab import BuildVocab
from imdb import ImdbDataset

import torch
from torch.utils.data import DataLoader
from model import TextCNN_Glove,TextCNN_FastText
from utils import train,test,device

if __name__ == '__main__':

    train_batch_size = 512
    test_batch_size = 128
    sequence_max_len = 100
    embedding_dim = 200


    imdb_dataset = BuildVocab()
    imdb_dataset.build_vocab(dataset = ImdbDataset(split='train'),min_count=1)
    
    imdb_dataset.make_dataset(dataset = ImdbDataset(split='train'),max_length=100)
    traindata = DataLoader(imdb_dataset, batch_size=train_batch_size, shuffle=True)
    
    imdb_dataset.make_dataset(dataset = ImdbDataset(split='test'),max_length=100)
    testdata = DataLoader(imdb_dataset, batch_size=test_batch_size,shuffle=True)
    
    # imdb_model = BiLSTM(input_size=imdb_dataset.vocab_size,
    #                     embed_size=embedding_dim).to(device())
    
    # imdb_model = TextCNN_Glove(vocab= imdb_dataset.text_vocab).to(device())
    imdb_model = TextCNN_FastText(vocab= imdb_dataset.text_vocab).to(device())
    path_textcnn_model = "./models/textcnn_model1.pkl"
    
    train(imdb_model, 10, train_dataloader=traindata, save_nam=path_textcnn_model,learning_rate=0.002)
    # 测试
    
    textcnn_model = torch.load(path_textcnn_model)
    test(textcnn_model,test_dataloader=testdata)
    
    # TextCNN_Glove
    # Epoch = 10,lr = 0.002
    # loss: 0.3407, 
    # Accuracy: 67746/78739(86.04%),
    # F1:0.87
    
    # TextCNN_FastText
    # Epoch = 10,lr = 0.002
    # loss: 0.3295, 
    # Accuracy: 68471/78739(86.96%),
    # F1:0.87
    
    