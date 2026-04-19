import os
import numpy as np
import pandas as pd
from datetime import datetime

def generate_date_col(start, end):
    time_series = pd.date_range(start=start, end=end, freq='h')
    return time_series.strftime('%Y-%m-%d %H:%M:%S')


def load_pre_processed_data():

    base_path = "../dataset/GEFCom2014/Load" 

    df_list = []

    for i in range(1, 16):
        
        folder_path = os.path.join(base_path, f"Task {i}")

        if not os.path.exists(folder_path):
            print(f"{folder_path} does not exist")
            continue
        
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('train.csv')]
        
        
        csv_path = os.path.join(folder_path, csv_files[0])
        
        try:
            df = pd.read_csv(csv_path)
            df_list.append(df)
            print(f"{csv_files[0]} data shape: {df.shape}")
            print(df.head(1))
            print(df.tail(1))
        except Exception as e:
            pass

    # concat all dataframes
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True).dropna()
        
        # save
        output_path = os.path.join(base_path, "Load.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"concatenated data shape: {combined_df.shape}")
    else:
        pass


def convert_Price_ot():
    base_path = "../dataset/GEFCom2014/Price"
    df = pd.read_csv(os.path.join(base_path, "Price.csv"))
    
    
    
    df = df.drop(df.columns[0], axis=1)
    df['timestamp'] = generate_date_col('2011-01-01 00:00:00', '2013-12-16 23:00:00')
    df.rename(columns={'timestamp':'date'}, inplace=True)
    df.rename(columns={'Zonal Price':'OT'}, inplace=True)
    
    print(df.head(5))
    print(df.shape)
    print(df.info())
    print(df.describe())
    
    df.to_csv(os.path.join(base_path, "Price_OT.csv"), index=False)


def convert_load_ot():
    base_path = "../dataset/GEFCom2014/Load" 

    df = pd.read_csv(os.path.join(base_path, "Load.csv"))
    
    
    
    df = df.drop(df.columns[0], axis=1)
    df['TIMESTAMP'] = generate_date_col('2005-01-01 01:00:00', '2011-12-01 00:00:00')
    df.rename(columns={'TIMESTAMP':'date'}, inplace=True)
    df.rename(columns={'LOAD':'OT'}, inplace=True)
    
    cols = df.columns.tolist()
    cols = cols[:1] + cols[2:] + [cols[1]]
    df = df[cols]
    print(df.head(5))
    print(df.shape)
    print(df.info())
    print(df.describe())
    df.to_csv(os.path.join(base_path, "Load_OT.csv"), index=False)
    # 保存


def convert_solar_ot():
    base_path = "../dataset/GEFCom2014/Solar" 

    for i in range(1, 4):
        df = pd.read_csv(os.path.join(base_path, f"Solarz{i}.csv"))
        
        df = df.drop(df.columns[0], axis=1)
        df['TIMESTAMP'] = generate_date_col('2012-04-01 01:00:00', '2014-07-01 00:00:00')
        df.rename(columns={'TIMESTAMP':'date'}, inplace=True)
        df.rename(columns={'POWER':'OT'}, inplace=True)
        

        print(df.head(5))
        print(df.shape)
        print(df.info())
        print(df.describe())
        df.to_csv(os.path.join(base_path, f"Solarz{i}_OT.csv"), index=False)
    # 保存

def convert_wind_ot():
    base_path = "../dataset/GEFCom2014/Wind" 

    for i in range(1, 11):
        df = pd.read_csv(os.path.join(base_path, f"Task15_W_Zone{i}.csv"))
        df = df.drop(df.columns[0], axis=1)
        df['TIMESTAMP'] = generate_date_col('2012-01-01 01:00:00', '2013-12-01 00:00:00')
        df.rename(columns={'TIMESTAMP':'date'}, inplace=True)
        df.rename(columns={'TARGETVAR':'OT'}, inplace=True)
        
        cols = df.columns.tolist()
        cols = cols[:1] + cols[2:] + [cols[1]]
        df = df[cols]
        print(df.head(5))
        print(df.shape)
        print(df.info())
        print(df.describe())
        df.to_csv(os.path.join(base_path, f"Windz{i}_OT.csv"), index=False)
        

if __name__ == '__main__':
    # load_pre_processed_data()
    # convert_load_ot()
    # convert_Price_ot()
    # convert_solar_ot()
    convert_wind_ot()