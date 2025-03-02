import pandas as pd
from sklearn.datasets import make_classification
import os


def create_data():
    features, target = make_classification(n_samples=2000,
                                           n_features=7,
                                           n_informative=5,
                                           n_redundant=1,
                                           n_classes=2,
                                           shuffle=True,
                                           random_state=42)
    feature_columns = ['feature_1',
                       'feature_2',
                       'feature_3',
                       'feature_4',
                       'feature_5',
                       'feature_6',
                       'feature_7']
    X = pd.DataFrame(data=features, columns=feature_columns)
    y = pd.Series(data=target, name='target')
    df = pd.concat((X, y), axis=1)
        
    test_size = int(df.shape[0]*0.30)
    test_data = df.sample(test_size, random_state=42).sort_index()
    train_data = df.drop(test_data.index).reset_index(drop=True)
    test_data.reset_index(drop=True, inplace=True)

    directories = ['train', 'test']
    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)
    
    train_data.to_csv('train/train_data.csv', sep=',', encoding='utf-8')
    test_data.to_csv('test/test_data.csv', sep=',', encoding='utf-8')   
    
create_data()
