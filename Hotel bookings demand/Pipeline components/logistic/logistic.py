import argparse
def lr(clean_data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression

    data = joblib.load(clean_data)
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    
    y_pred = log_reg.predict(X_test)
    
    train = log_reg.score(X_train, y_train)
    test = log_reg.score(X_test, y_test)
    
    acc_reg = accuracy_score(y_test, y_pred)
    print(acc_reg)
    
    report_classification = classification_report(y_test, y_pred)
    print(classification_report(y_test,y_pred))
    
    lr_metrics = {'train':train, 'test':test, 'accuracy':acc_reg, 'report':report_classification, 'model':log_reg}
    joblib.dump(lr_metrics,'lr_metrics')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data')
    args = parser.parse_args()
    lr(args.clean_data)   

