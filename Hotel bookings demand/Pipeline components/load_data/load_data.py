def load_data():
    import joblib
    import pandas as pd
    
    df = pd.read_csv('https://github.com/charlesa101/KubeflowUseCases/blob/draft/Hotel%20bookings%20demand/hotel_bookings.csv?raw=true')
    joblib.dump(df, 'data')
    
if __name__ == '__main__':
    load_data()

