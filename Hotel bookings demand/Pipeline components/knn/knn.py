import argparse
def knn(clean_data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from sklearn.linear_model import KNeighborsClassifier

    data = joblib.load(clean_data)
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']
    
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    train = knn.score(X_train, y_train)
    test = knn.score(X_test, y_test)
    
    acc_knn = accuracy_score(y_test, y_pred)
    print(acc_knn)
    
    report_classification = classification_report(y_test, y_pred)
    print(classification_report(y_test,y_pred))
    
    knn_metrics = {'train':train, 'test':test, 'accuracy':acc_knn, 'report':report_classification, 'model':knn}
    joblib.dump(knn_metrics,'knn_metrics')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data')
    args = parser.parse_args()
    knn(args.clean_data)   