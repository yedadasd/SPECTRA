import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df_raw = []
    with open(os.path.join('../dataset/Solar', 'solar_AL.txt'), "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            data_line = np.stack([float(i) for i in line])
            df_raw.append(data_line)
    df_raw = np.stack(df_raw, 0)
    df_raw = pd.DataFrame(df_raw)
    
    
    print(df_raw.info())
    print(df_raw.describe())
    print(df_raw.head())