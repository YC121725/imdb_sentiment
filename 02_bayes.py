from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

from imdb import ImdbDataset
import numpy as np
import evaluate
import time 
if __name__ == '__main__':

    a = time.time()
    traindataset = ImdbDataset()
    # testdataset = ImdbDataset(split='test')
    
    all_text = [i[0] for i in traindataset ]
    all_label = [i[1] for i in traindataset ]
    vectorizer = TfidfVectorizer(max_features=5000,lowercase=True)

    corpusTotoken_count = vectorizer.fit_transform(all_text).todense()
    X_data = np.array(corpusTotoken_count)
    Y_data = np.array(all_label)
    
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.3)

    model = MultinomialNB()
    model2 = ComplementNB()
    model.fit(x_train,y_train)
    # model2.fit(x_train,y_train)
    predictions_GB = model.predict(x_test)
    # predictions_GB2 = model2.predict(x_test)

    # print(predictions_GB.shape)
    # acc_num = 0
    # acc_num2 = 0
    # for i in range(len(predictions_GB)):
    #     if predictions_GB[i] == y_test[i]:
    #         acc_num +=1
    # for i in range(len(predictions_GB2)):
    #     if predictions_GB2[i] == y_test[i]:
    #         acc_num2 +=1
    
    # print(acc_num/len(predictions_GB))
    # # 0.86413
    # print(recall_score(y_test,predictions_GB))
    # print(acc_num2/len(predictions_GB2))
    # print(recall_score(y_test,predictions_GB2))
    
    acc = evaluate.load('accuracy')
    # acc2 = evaluate.load('accuracy')
    f1 = evaluate.load('f1')
    # f1_ = evaluate.load('f1')
    
    print("predict")

    acc.add_batch(predictions = predictions_GB,
                  references=y_test)
    f1.add_batch(predictions = predictions_GB,
                  references=y_test)
    
    # acc2.add_batch(predictions = predictions_GB2,
    #               references=y_test)
    # f1_.add_batch(predictions = predictions_GB2,
    #               references=y_test)
    b = time.time()
    
    print(b-a)  
    print(acc.compute())
    print(f1.compute())
    
    # print(acc2.compute())
    # print(f1_.compute())
    
    # # 39s，41s 45s
    # 40s 左右
    # {'accuracy': 0.8526666666666667}
    # {'f1': 0.8521739130434783}
    # {'accuracy': 0.853}
    # {'f1': 0.8521820741435947}