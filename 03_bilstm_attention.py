from vocab import BuildVocab
from imdb import ImdbDataset
import torch
from torch.utils.data import DataLoader

from model import BiLSTM
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
    
    imdb_model = BiLSTM(input_size=imdb_dataset.vocab_size,
                        embed_size=embedding_dim).to(device())
    
    
    path_lstm_model = "./models/lstm_model.pkl"
    train(imdb_model, 10,save_nam= path_lstm_model,train_dataloader=traindata)
    # 测试
    lstm_model = torch.load(path_lstm_model)
    test(lstm_model,test_dataloader=testdata)
    
    # Epoch 10
    # Avg. loss: 0.0485, 
    # Accuracy: 77321/78739(98.20%),
    # F1:0.98
    