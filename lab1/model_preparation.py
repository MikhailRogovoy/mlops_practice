import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


def prepare_model():
    train_data = pd.read_csv('train/train_data_scaled.csv', sep=',', encoding='utf-8', index_col=0)    
    y_train = train_data['target']
    X_train = train_data.drop(labels='target', axis=1)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    pkl_filename = "model.pkl" 
    with open(pkl_filename, 'wb') as file: 
        pickle.dump(model, file)

prepare_model()