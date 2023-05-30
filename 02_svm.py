from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm

from imdb import ImdbDataset
import numpy as np
import evaluate
import time

if __name__ == '__main__':
    a = time.time()
    traindataset = ImdbDataset()
    
    all_text = [i[0] for i in traindataset ]
    all_label = [i[1] for i in traindataset ]

    vectorizer = TfidfVectorizer(max_features=5000,lowercase=True)

    corpusTotoken_count = vectorizer.fit_transform(all_text).todense()
    X_data = np.array(corpusTotoken_count)
    Y_data = np.array(all_label)
    
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.3)
    
    # print(len(x_train))
    # print(len(y_train))

    model = svm.SVC(verbose=True)
    model.fit(x_train,y_train)
    predictions_GB = model.predict(x_test)

    acc = evaluate.load('accuracy')
    f1 = evaluate.load('f1')
    
    print("predict")

    acc.add_batch(predictions = predictions_GB,
                  references=y_test)
    f1.add_batch(predictions = predictions_GB,
                  references=y_test)
    b = time.time()
    print(acc.compute())
    print(f1.compute())
    print(b-a)
    # optimization finished, #iter = 24539
    # obj = -8012.610177, rho = 0.049055
    # nSV = 17758, nBSV = 7642
    # Total nSV = 17758
    # [LibSVM]predict
    # {'accuracy': 0.8927333333333334}
    # {'f1': 0.8939144194633085}
    # 3970.048374891281 s
