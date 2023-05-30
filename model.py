import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe,FastText
from transformers import BertModel

class BiLSTM(nn.Module):
    def __init__(self,input_size,embed_size):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embed_size, padding_idx = 1)
        # 设置了batch first
        self.lstm = nn.LSTM(input_size=200, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.5)
        
        self.fc1 = nn.Linear(100, 1)
        self.fc2 = nn.Linear(100, 2)
        
    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """
        # print(input.shape)
        # (512, 100)
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        output, (h_n, c_n) = self.lstm(input_embeded)  # h_n :[4,batch_size,hidden_size]
        
        H = output[:,:,:100] + output[:,:,100:]        # 
        
        alpha = F.relu(self.fc1(H))
        out = F.tanh(torch.matmul(H.transpose(1,2),alpha)).squeeze(2)
        output = F.log_softmax(self.fc2(out),dim=-1)
        
        return output

class TextCNN_Glove(nn.Module):
    r"""
    Args:
        vocab (torchtext.vocav.Vocab): word dictionary, indexed by vocab
        embed_size (int): number of embedding. Default 100, chossible values are 50; 100; 200; 300
        kernel_size (list(int)): 每一个卷积核大小的宽 长等于 embed_size
        num_channels (int): 每个卷积核卷积后输出的channel 大小
    
    __init__:
        导入glove预训练词向量，
        
    """
    def __init__(self, vocab, embed_size = 300, kernel_sizes=[3, 4, 5, 6, 7,8], num_channels = 3, out_class = 2):
        super(TextCNN_Glove,self).__init__()
        
        # 预训练词向量
        self.glove = GloVe(name="6B", dim = embed_size)
        self.fasttext = FastText()
        
        # 训练时候，参数可学习
        self.unfrozen_embedding = nn.Embedding.from_pretrained(
            self.glove.get_vecs_by_tokens(vocab.get_itos()), padding_idx=vocab["<PAD>"]
        )

        # 训练时候，参数不可学习
        self.frozen_embedding = nn.Embedding.from_pretrained(
            self.glove.get_vecs_by_tokens(vocab.get_itos()), padding_idx=vocab["<PAD>"], freeze=True,
        )
        
        # 卷积层
        self.convs_for_unfrozen = nn.ModuleList()
        self.convs_for_frozen = nn.ModuleList()
        
        
        for kernel in kernel_sizes:
            self.convs_for_unfrozen.append(
                nn.Conv2d(in_channels = 1,
                        out_channels = num_channels,
                        kernel_size = (kernel, embed_size),
                ))
            
            self.convs_for_frozen.append(
                nn.Conv2d(
                    in_channels = 1,
                    out_channels = num_channels,
                    kernel_size = (kernel, embed_size),
                ))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * 12, out_class)
        self.apply(self._init_weights)

    def forward(self, x):
        x = x.unsqueeze(1)                      # (batch_size, seq_len) -> (batch_size, channel, seq_len)

        x_unfrozen = self.unfrozen_embedding(x) # (batch_size, channel, seq_len, embed_size) -> (batch_size,
        x_frozen = self.frozen_embedding(x)      
        
        # # 卷积
        conved_frozen = [self.relu(conv(x_frozen)) for conv in self.convs_for_frozen]
        conved_unfrozen = [self.relu(conv(x_unfrozen)) for conv in self.convs_for_unfrozen]
        
        # 池化
        max_pooled_frozen = [nn.MaxPool2d((conved.shape[2],1))(conved).squeeze().squeeze()  for conved in conved_frozen] 
        
        max_pooled_unfrozen = [nn.MaxPool2d((conved.shape[2],1))(conved).squeeze().squeeze()   for conved in conved_unfrozen] 
         
        # # 将向量拼接起来后得到一个更长的向量
        feature_vector = torch.cat(
            max_pooled_frozen + max_pooled_unfrozen, dim=-1)        # (batch_size, 600)
        
        
        output = self.fc(self.dropout(feature_vector))              # (batch_size, 2)
        return F.log_softmax(output,dim=-1)

    def _init_weights(self, m):
        # 仅对线性层和卷积层进行xavier初始化
        if type(m) in (nn.Linear, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)


class TextCNN_FastText(nn.Module):
    r"""
    Args:
        vocab (torchtext.vocav.Vocab): word dictionary, indexed by vocab
        embed_size (int): number of embedding. Default 100, chossible values are 50; 100; 200; 300
        kernel_size (list(int)): 每一个卷积核大小的宽 长等于 embed_size
        num_channels (int): 每个卷积核卷积后输出的channel 大小
    
    __init__:
        导入glove预训练词向量，
        
    """
    def __init__(self, vocab, embed_size = 300, kernel_sizes=[3, 4, 5, 6, 7,8], num_channels = 3, out_class = 2):
        super(TextCNN_FastText,self).__init__()
        
        # 预训练词向量
        self.fasttext = FastText()
        
        # 训练时候，参数可学习
        self.unfrozen_embedding = nn.Embedding.from_pretrained(
            self.fasttext.get_vecs_by_tokens(vocab.get_itos()), padding_idx=vocab["<PAD>"]
        )

        # 训练时候，参数不可学习
        self.frozen_embedding = nn.Embedding.from_pretrained(
            self.fasttext.get_vecs_by_tokens(vocab.get_itos()), padding_idx=vocab["<PAD>"], freeze=True,
        )
        
        # 卷积层
        self.convs_for_unfrozen = nn.ModuleList()
        self.convs_for_frozen = nn.ModuleList()
        
        
        for kernel in kernel_sizes:
            self.convs_for_unfrozen.append(
                nn.Conv2d(in_channels = 1,
                        out_channels = num_channels,
                        kernel_size = (kernel, embed_size),
                ))
            
            self.convs_for_frozen.append(
                nn.Conv2d(
                    in_channels = 1,
                    out_channels = num_channels,
                    kernel_size = (kernel, embed_size),
                ))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * 12, out_class)
        self.apply(self._init_weights)

    def forward(self, x):
        x = x.unsqueeze(1)                      # (batch_size, seq_len) -> (batch_size, channel, seq_len)

        x_unfrozen = self.unfrozen_embedding(x) # (batch_size, channel, seq_len, embed_size) -> (batch_size,
        x_frozen = self.frozen_embedding(x)      
        
        # # 卷积
        conved_frozen = [self.relu(conv(x_frozen)) for conv in self.convs_for_frozen]
        conved_unfrozen = [self.relu(conv(x_unfrozen)) for conv in self.convs_for_unfrozen]
        
        # 池化
        max_pooled_frozen = [nn.MaxPool2d((conved.shape[2],1))(conved).squeeze().squeeze()  for conved in conved_frozen] 
        
        max_pooled_unfrozen = [nn.MaxPool2d((conved.shape[2],1))(conved).squeeze().squeeze()   for conved in conved_unfrozen] 
         
        # # 将向量拼接起来后得到一个更长的向量
        feature_vector = torch.cat(
            max_pooled_frozen + max_pooled_unfrozen, dim=-1)        # (batch_size, 600)
        
        
        output = self.fc(self.dropout(feature_vector))              # (batch_size, 2)
        return F.log_softmax(output,dim=-1)

    def _init_weights(self, m):
        # 仅对线性层和卷积层进行xavier初始化
        if type(m) in (nn.Linear, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

class MyBert(nn.Module):
    r"""
    # Introduction
    ------------------
        基于 huggingface 的 transformers 库的bert模型
    
    # Description
        1. 使用BertModel.from_pretrained 函数导入预训练模型，训练过程中模型的参数梯度为 False
    """
    def __init__(self):
        super(MyBert, self).__init__()
        self.fc = nn.Linear(768, 2)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(device)
        self.pretrained = BertModel.from_pretrained("bert-base-uncased",cache_dir='./.bert_cache').to(device)
        # 需要移动到cuda上
        # 不训练,不需要计算梯度
        # for param in self.pretrained.parameters():
        #     param.requires_grad_(True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # with torch.no_grad():
        #     out = self.pretrained(
        #         input_ids = input_ids,
        #         attention_mask = attention_mask,
        #         token_type_ids = token_type_ids,
        #     )
        out = self.pretrained(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out
