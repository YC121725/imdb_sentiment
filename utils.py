import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train(imdb_model,epoch,train_dataloader,save_nam:str,learning_rate=1e-3):
    """
    :param imdb_model:
    :param epoch:
    :return:
    """
    optimizer = Adam(imdb_model.parameters(),lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            # loss = criterion(output, target)
            loss = F.nll_loss(output, target.long())
            loss.backward()
            optimizer.step()
            bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))
    # 保存模型
    path_model = save_nam
    torch.save(imdb_model, path_model)
    # 保存模型参数
    # path_state_dict = "./models/textcnn_model_state_dict.pkl"
    # net_state_dict = imdb_model.state_dict()
    # torch.save(net_state_dict, path_state_dict)


def test(imdb_model,test_dataloader):
    """
    验证模型
    :param imdb_model:
    :return:
    """
    test_loss = 0
    correct = 0
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    imdb_model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
            acc.add_batch(predictions = pred,
                         references = target)
            f1.add_batch(predictions = pred,
                         references = target)
    test_loss /= len(test_dataloader.dataset)

    
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{}({:.2f}%),F1:{:.2f} \n'.format(
        test_loss, correct ,len(test_dataloader.dataset),
        acc.compute()['accuracy']*100., f1.compute()['f1']))