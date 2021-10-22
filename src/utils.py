import pandas as pd
import numpy as np


def trendline(data:pd.Series, order=1):
    coeffs = np.polyfit(data.index.values, list(data.values), order)
    slope = coeffs[-2]
    return float(slope)


def create_lag_features(df:pd.DataFrame, columns, lags_start: int, lags_end: int):
    for col in columns:
        for lag in range(lags_start, lags_end + 1):
            lag_feature_name = col + f'_lag{lag}'
            df[lag_feature_name] = df.groupby('breath_id')[col].shift(lag)
            df[lag_feature_name].fillna(0, inplace=True)

def create_windows_features(df:pd.DataFrame, columns, windows: list):
    for col in columns:
        print(col)
        for window in windows:
            print('window', window)
            new_columns = [f'{col}_window_{window}_mean',
            f'{col}_window_{window}_sum', f'{col}_window_{window}_min', f'{col}_window_{window}_max']
            df[new_columns] = (df.groupby('breath_id')[col].rolling(window=window,min_periods=windows[0])
                                                              .agg({f'{col}_window_{window}_mean':"mean",
                                                                    f'{col}_window_{window}_sum':"sum",
                                                                    f'{col}_window_{window}_min':"min",
                                                                    f'{col}_window_{window}_max':"max"})
                                                               .reset_index(level=0,drop=True))
            print('finished')
            print('*' * 50)



def add_features(df, windows):
    df['cross']= df['u_in'] * df['u_out']
    df['cross2']= df['time_step'] * df['u_out']
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    print("Step-1...Completed")
    
    create_lag_features(df, ['u_in', 'u_out'], lags_start=-4, lags_end=4)
    print("Step-2(lags creation) Completed") 
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_in__mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    print("Step-3...Completed")
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    print("Step-4...Completed")
    
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] /df['count']
    
    df['breath_id_lag']=df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2']=df['breath_id'].shift(2).fillna(0)
    df['breath_id_lagsame']=np.select([df['breath_id_lag']==df['breath_id']],[1],0)
    df['breath_id_lag2same']=np.select([df['breath_id_lag2']==df['breath_id']],[1],0)
    df['breath_id__u_in_lag'] = df['u_in'].shift(1).fillna(0)
    df['breath_id__u_in_lag'] = df['breath_id__u_in_lag'] * df['breath_id_lagsame']
    df['breath_id__u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['breath_id__u_in_lag2'] = df['breath_id__u_in_lag2'] * df['breath_id_lag2same']
    print("Step-5...Completed")
    
    df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['ewm_u_in_mean'] = (df\
                           .groupby('breath_id')['u_in']\
                           .ewm(halflife=9)\
                           .mean()\
                           .reset_index(level=0,drop=True))
    create_windows_features(df, ['u_in', 'u_out'], windows)
    print("Step-6...Completed")
    
    df['u_in_lagback_diff1'] = df['u_in'] - df['u_in_lag-1']
    df['u_out_lagback_diff1'] = df['u_out'] - df['u_out_lag-1']
    df['u_in_lagback_diff2'] = df['u_in'] - df['u_in_lag-2']
    df['u_out_lagback_diff2'] = df['u_out'] - df['u_out_lag-2']
    print("Step-7...Completed")
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")
    
    return df
