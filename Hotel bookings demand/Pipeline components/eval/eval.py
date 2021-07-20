import argparse
def eval(rfc_metrics, keras_metrics, lr_metrics, knn_metrics):
    import joblib
    import pandas as pd
    import pprint
    
    rfc_metrics=joblib.load(rfc_metrics)
    keras_metrics=joblib.load(keras_metrics)
    lr_metrics=joblib.load(lr_metrics)
    knn_metrics=joblib.load(knn_metrics)
    
    lis = [rfc_metrics, keras_metrics, lr_metrics, knn_metrics]
    max = 0
    for i in lis:
        accuracy = i['test']
        if accuracy > max:
            max = accuracy
            model = i['model']
            metrics = i
  
    print('Best metrics \n\n')
    pprint.pprint(metrics)

    joblib.dump(model, 'best_model')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rfc_metrics')
    parser.add_argument('--keras_metrics')
    parser.add_argument('--lr_metrics')
    parser.add_argument('--knn_metrics')
    args = parser.parse_args()
    eval(args.rfc_metrics, args.keras_metrics, args.lr_metrics, args.knn_metrics)

