import argparse
def preprocessing(data):
    import joblib
    import numpy as np
    import pandas as pd
    import datetime
    from sklearn.model_selection import train_test_split as tts
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    
    df = joblib.load(data)
    
    # drop missing values
    df = df.drop(['company', 'agent'], axis=1)
    df = df.dropna(subset=['country', 'children'], axis=0)
    df = df.reset_index(drop=True)
    
    # Converting wrong datatype columns to correct type (object to datetime)
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    
    # Converting string month to numerical one (Dec = 12, Jan = 1, etc.)
    datetime_object = df['arrival_date_month'].str[0:3]
    month_number = np.zeros(len(datetime_object))
    
    # Creating a new column based on numerical representation of the months
    for i in range(0, len(datetime_object)):
        datetime_object[i] = datetime.datetime.strptime(datetime_object[i], "%b")
        month_number[i] = datetime_object[i].month
    
    # Float to integer conversion
    month_number = pd.DataFrame(month_number).astype(int)
    
    # 3 columns merged into one
    df['arrival_date'] = df['arrival_date_year'].map(str) + '-' + month_number[0].map(str) + '-' + df['arrival_date_day_of_month'].map(str)
    
    # Dropping already used columns
    df = df.drop(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
                  'arrival_date_week_number'], axis=1)
    
    # convert the newly created arrival_date feature to datetime type
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    
    
    # Calculating total guests by combining adults, children and babies columns
    df['total guests'] = df['adults'] + df['children'] + df['babies']
    
    # drop data points that include zero Total Guests
    df = df[df['total guests'] != 0]
    
    # Total Number of Days Stayed
    df['total stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    dataNoCancel = df[df['is_canceled'] == 0]
    dataNoCancel = dataNoCancel.reset_index(drop=True)
    
    df = df.drop(['adults', 'children', 'babies',
                  'stays_in_weekend_nights',
                  'stays_in_week_nights', 'arrival_date', 
                  'reservation_status_date'], axis=1)
    
    # Categorical variables preprocessing with label encoding
    list_1 = list(df.columns)
    cate_list=[]
    for i in list_1:
    if df[i].dtype=='object':
        cate_list.append(i)
        
    # transform the categorical variables with label encoder
    le = LabelEncoder()
    for i in cate_list:
        df[i] = le.fit_transform(df[i])
        
    # split the data into dependent variables and independent variable
    X = df.drop(['hotel'],axis=1)
    y = df.hotel
    
    # split the data into training and test set
    X_train,X_test,y_train,y_test = tts(X,y,random_state=36,test_size=0.3)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    data_dic = {"X_train": X_train,"X_test": X_test, "Y_train": y_train, "Y_test": y_test}
    
    joblib.dump(data_dic, 'clean_data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  preprocessing(args.data)  

