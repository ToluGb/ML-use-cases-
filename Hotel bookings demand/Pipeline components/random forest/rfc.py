import argparse
def rfc(clean_data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier

    data = joblib.load(clean_data)
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']
    
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    
    y_pred = rfc.predict(X_test)
    
    train = rfc.score(X_train, y_train)
    test = rfc.score(X_test, y_test)
    
    acc_rfc = accuracy_score(y_test, y_pred)
    print(acc_rfc)
    
    report_classification = classification_report(y_test, y_pred)
    print(classification_report(y_test,y_pred))
    
    rfc_metrics = {'train':train, 'test':test, 'accuracy':acc_rfc, 'report':report_classification, 'model':rfc}
    joblib.dump(rfc_metrics,'rfc_metrics')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data')
    args = parser.parse_args()
    rfc(args.clean_data)   

