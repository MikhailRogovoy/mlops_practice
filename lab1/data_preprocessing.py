import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_data():
    train_data = pd.read_csv('train/train_data.csv', sep=',', encoding='utf-8', index_col=0)
    test_data = pd.read_csv('test/test_data.csv', sep=',', encoding='utf-8', index_col=0)
    y_train = train_data['target']
    X_train = train_data.drop(labels='target', axis=1)
    y_test = test_data['target']
    X_test = test_data.drop(labels='target', axis=1)
  
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y=y_train)
    X_test_scaled = scaler.fit_transform(X_test, y=y_test)

    feature_columns = ['feature_1',
                       'feature_2',
                       'feature_3',
                       'feature_4',
                       'feature_5',
                       'feature_6',
                       'feature_7']
    X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=feature_columns)
    
    train_data_scaled = pd.concat((X_train_scaled, y_train), axis=1)
    test_data_scaled = pd.concat((X_test_scaled, y_test), axis=1)

    train_data_scaled.to_csv('train/train_data_scaled.csv', sep=',', encoding='utf-8')
    test_data_scaled.to_csv('test/test_data_scaled.csv', sep=',', encoding='utf-8')

scale_data()