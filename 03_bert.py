from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import AdamW, get_scheduler

import evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm
from model import MyBert
from imdb import ImdbDataset

from peft import LoraConfig, get_peft_model
# 导入数据集

imdb_dataset = ImdbDataset.from_dataset()

# 导入预训练模型
ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(ckpt)

def tokenize(exmaple):
    return tokenizer(exmaple['text'],padding=True,truncation=True)

tokenized_dataset = imdb_dataset.map(tokenize)
tokenized_dataset = tokenized_dataset.remove_columns('text')
# print(tokenized_dataset)

collate_fn = DataCollatorWithPadding(tokenizer)

train_dataloader = DataLoader(dataset=tokenized_dataset['train'],
                              shuffle=True,
                              batch_size=8,
                              collate_fn=collate_fn)

test_dataloader = DataLoader(dataset=tokenized_dataset['test'],
                             shuffle=True,
                             batch_size=8,
                             collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

peft_config = LoraConfig(task_type = 'SEQ_CLS',inference_mode=False,r=8,lora_alpha=32,lora_dropout=0.1)
# model = MyBert().to(device)
model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
model = get_peft_model(model,peft_config)

optimizer = AdamW(model.parameters(), lr=5e-5)
# criterion = nn.CrossEntropyLoss()
num_epochs = 3
num_train_steps = num_epochs*len(train_dataloader)

lr_scheduler = get_scheduler('linear', 
                optimizer = optimizer,
                num_warmup_steps=0,
                num_training_steps=num_train_steps)

p_bar = tqdm(range(num_train_steps))

model.train()
scaler = GradScaler()

for epoch in range(num_epochs):
  # 'labels', 'input_ids', 'token_type_ids', 'attention_mask'
    for batch in train_dataloader:
        batch = batch.to(device)
        
        # NOTE 增加 混合精度训练 
        ''' 时间缩短50% '''
        
        with autocast():
            out = model(input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                labels = batch['labels']
                )

            # loss = criterion(out, batch['labels'])
            loss = out.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #################################
        
        lr_scheduler.step()
        optimizer.zero_grad()
        p_bar.update(1)
        # out = model(input_ids=batch['input_ids'],
        #         attention_mask=batch['attention_mask'],
        #         token_type_ids=batch['token_type_ids'])

        # loss = criterion(out, batch['labels'])    
        # loss.backward()
        # optimizer.step()
        # 
        # lr_scheduler.step()
        # optimizer.zero_grad()
        # p_bar.update(1)

acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")
model.eval()

for batch in test_dataloader:
    batch = batch.to(device)
    with torch.no_grad():
        out = model(input_ids=batch['input_ids'],
               attention_mask=batch['attention_mask'],
               token_type_ids=batch['token_type_ids'],
               labels=batch['labels'])
    logits = out.logits
    
    predictions = torch.argmax(logits, dim=-1)
    acc.add_batch(predictions = predictions,
           references = batch['labels'])
    f1.add_batch(predictions = predictions,
           references = batch['labels'])

print('acc:\t',acc.compute())
print('f1:\t',f1.compute())

# Time 1:52:59
# acc:	 {'accuracy': 0.9324}
# f1:	 {'f1': 0.9328031809145129}

# LoRA+混合精度
# Time 44:13
# acc:	 {'accuracy': 0.92236}
# f1:    {'f1': 0.9225860487376859}