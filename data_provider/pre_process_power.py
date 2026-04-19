import pandas as pd
import numpy as np
import os
from netCDF4 import Dataset


def get_data(path_template):
    print("Loading data: ", path_template)
    dataset = Dataset(path_template, mode='r')
    channel = dataset.variables["channel"][:]
    data = dataset.variables["data"][:]
    mean_values = np.array([np.mean(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
    return pd.DataFrame(mean_values, columns=channel)


def pre_process_power_1(station):
    train_path = f"../dataset/NewEnergy/初赛训练集/fact_data/{station}_normalization_train.csv"
    data = pd.read_csv(train_path)
    data = data[96:]
    data['时间'] = pd.to_datetime(data['时间'], format='%Y-%m-%d %H:%M:%S')
    # data.set_index('时间', inplace=True)
    
    return data

def pre_process_weather_1(station):
    """
    preProcess for single New energy power stations.
    """
    train_path_template = "../dataset/NewEnergy/初赛训练集/nwp_data_train/{}/{}/{}.nc"
    
    date_range = pd.date_range(start='2024-01-01', end='2024-12-30')
    date_range = [date.strftime('%Y%m%d') for date in date_range]
    
    nwp_1_data = pd.concat([get_data(train_path_template.format(station, 'NWP_1', str(date))) for date in date_range], axis=0).reset_index(drop=True)
    nwp_2_data = pd.concat([get_data(train_path_template.format(station, 'NWP_2', str(date))) for date in date_range], axis=0).reset_index(drop=True)
    nwp_3_data = pd.concat([get_data(train_path_template.format(station, 'NWP_3', str(date))) for date in date_range], axis=0).reset_index(drop=True)
    # data = pd.concat(data, axis=0).reset_index(drop=True)
    c_nwp_1_3_data = nwp_1_data.columns
    c_nwp_2_data = nwp_2_data.columns
    features_common = list(set(c_nwp_1_3_data) & set(c_nwp_2_data))
    features_1_3 = list(set(c_nwp_1_3_data) - set(features_common))
    features_2 = list(set(c_nwp_2_data) - set(features_common))
    
    average_data = pd.DataFrame()
    for feature in features_common:
        values = [nwp_1_data[feature], nwp_2_data[feature], nwp_3_data[feature]]
        average_data[feature] = np.mean(values, axis=0)
    
    # sp
    for feature in features_1_3:
        values = [nwp_1_data[feature], nwp_3_data[feature]]
        average_data[feature] = np.mean(values, axis=0)
    
    # msl
    for feature in features_2:
        average_data[feature] = nwp_2_data[feature]
    
    start_time = pd.Timestamp('2024-01-02')
    weather_time_index = start_time + pd.to_timedelta(average_data.index, unit='h')
    average_data.index = weather_time_index
    average_data = average_data.resample('15min').interpolate() 
    additional_data = pd.DataFrame([average_data.iloc[-1]] * 3, index=pd.date_range(start='2024-12-31-23:15:00', end='2024-12-31-23:45:00', freq='15min'))
    data = pd.concat([average_data, additional_data], axis=0)

    return data

    


if __name__ == '__main__':
    stations = np.arange(1, 11)
    # wind
    wind_mean = []
    solar_mean = []
    # for station in stations[:5]:
    #     data_weather = pre_process_weather_1(station)
    #     data_power = pre_process_power_1(station)
    #     data_power.index = data_weather.index
    #     data = pd.concat([data_weather, data_power], axis=1)
        
    #     columns = list(data.columns)
    #     columns = [columns[-2]] + columns[:-2] + [columns[-1]]
    #     data = data[columns].rename(columns={'时间': 'date', '功率(MW)': 'OT'})
        
    #     data.to_csv(f"../dataset/NewEnergy/初赛训练集/wind{station}.csv", index=False)
    # # solar
    # for station in stations[5:]:
    #     data_weather = pre_process_weather_1(station)
    #     data_power = pre_process_power_1(station)
    #     data_power.index = data_weather.index
    #     data = pd.concat([data_weather, data_power], axis=1)
        
    #     columns = list(data.columns)
    #     columns = [columns[-2]] + columns[:-2] + [columns[-1]]
    #     data = data[columns].rename(columns={'时间': 'date', '功率(MW)': 'OT'})
        
    #     data.to_csv(f"../dataset/NewEnergy/初赛训练集/solar{station}.csv", index=False)
    
    # for station in stations:
    #     if station <= 5:
    #         data = pd.read_csv(f"../dataset/NewEnergy/初赛训练集/wind{station}.csv")
    #         wind_mean.append(data)
     
                        
    #     else:
    #         data = pd.read_csv(f"../dataset/NewEnergy/初赛训练集/solar{station}.csv")
    #         solar_mean.append(data)

    # time_column = solar_mean[0].iloc[:, 0]

    # value_dfs = [df.iloc[:, 1:] for df in solar_mean]

    # sum_df = value_dfs[0]
    # for df in value_dfs[1:]:
    #     sum_df = sum_df.add(df)

    # mean_values = sum_df / len(solar_mean)

    # result_df = pd.concat([time_column, mean_values], axis=1)

    # result_df.columns = solar_mean[0].columns
    
    # result_df.to_csv(f"../dataset/NewEnergy/初赛训练集/solar_mean.csv", index=False)

    # value_dfs = [df.iloc[:, 1:] for df in wind_mean]

    # sum_df = value_dfs[0]
    # for df in value_dfs[1:]:
    #     sum_df = sum_df.add(df)

    # mean_values = sum_df / len(wind_mean)

    # result_df = pd.concat([time_column, mean_values], axis=1)

    # result_df.columns = wind_mean[0].columns
    
    # result_df.to_csv(f"../dataset/NewEnergy/初赛训练集/wind_mean.csv", index=False)

    # result_df = pd.concat([df.set_index('time') for df in dfs], axis=1, keys=range(len(dfs)))
    # result_df = result_df.groupby(level=1, axis=1).mean()
    # result_df.reset_index(inplace=True)
