import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

def test_model():
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'rb') as file: 
        model = pickle.load(file)

    test_data = pd.read_csv('test/test_data_scaled.csv', sep=',', encoding='utf-8', index_col=0)    
    y_test = test_data['target']
    X_test = test_data.drop(labels='target', axis=1)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred) 
    print(f"Model test accuracy is: {score:.3f}")

test_model()