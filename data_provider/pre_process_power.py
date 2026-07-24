import pandas as pd
import numpy as np
import os
from netCDF4 import Dataset


def get_data(path_template):
    print("Loading data: ", path_template)
    dataset = Dataset(path_template, mode='r')
    channel = dataset.variables["channel"][:]
    data = dataset.variables["data"][:]

    # Handle NetCDF masked arrays / fill values
    if hasattr(data, 'mask'):
        data = data.filled(np.nan)

    mean_values = np.array([np.mean(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
    df = pd.DataFrame(mean_values, columns=channel)

    # Forward-fill then backward-fill any NaN (fill values in spatial mean)
    if df.isnull().any().any():
        n_before = df.isnull().sum().sum()
        df = df.ffill().bfill()
        print(f"  Fixed {n_before} NaN values (fill values) in {path_template}")

    return df


def pre_process_power_1(station):
    train_path = f"../dataset/NewEnergy/fact_data/{station}_normalization_train.csv"
    data = pd.read_csv(train_path)
    data = data[96:]

    # --- clean power data ---
    pw_col = '功率(MW)'

    # 1. Clip negative power to 0 (physically impossible, likely rounding artifacts)
    neg_count = (data[pw_col] < 0).sum()
    if neg_count > 0:
        print(f"  Station {station}: clipping {neg_count} negative power values to 0")
        data[pw_col] = data[pw_col].clip(lower=0)

    # 2. Forward-fill then backward-fill nulls (sensor dropouts)
    null_count = data[pw_col].isnull().sum()
    if null_count > 0:
        print(f"  Station {station}: filling {null_count} null power values (ffill → bfill)")
        data[pw_col] = data[pw_col].ffill().bfill()

    data['时间'] = pd.to_datetime(data['时间'], format='%Y-%m-%d %H:%M:%S')

    return data

def pre_process_weather_1(station):
    """
    preProcess for single New energy power stations.
    (Original version: averages three NWP sources together.)
    """
    train_path_template = "../dataset/NewEnergy/nwp_data_train/{}/{}/{}.nc"

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

    # Fill any NaN at edges that interpolation couldn't reach
    if average_data.isnull().any().any():
        n_edge = average_data.isnull().sum().sum()
        average_data = average_data.ffill().bfill()
        print(f"  Fixed {n_edge} edge NaN after resample+interpolate")

    additional_data = pd.DataFrame([average_data.iloc[-1]] * 3, index=pd.date_range(start='2024-12-31-23:15:00', end='2024-12-31-23:45:00', freq='15min'))
    data = pd.concat([average_data, additional_data], axis=0)

    return data


def pre_process_weather_v2(station, keep_ensemble_stats=False):
    """
    Improved preprocessing that keeps each NWP source as a SEPARATE feature,
    rather than averaging them together.

    Rationale:
    - NWP_1, NWP_2, NWP_3 are different numerical weather prediction models.
      Averaging them loses the ensemble spread (model disagreement), which is
      itself a valuable predictor of forecast uncertainty.
    - ML models can learn optimal source combinations — including which source
      to trust under which conditions — but only if the per-source signals are
      preserved.
    - When models disagree widely, the downstream forecaster can learn to be
      more conservative; when they agree, more confident.

    Feature mapping:
    - Common (ghi, poai, t2m, tcc, tp, u100, v100): 7 features * 3 sources = 21 cols
    - sp (NWP_1 & NWP_3 only): 1 feature * 2 sources = 2 cols
    - msl (NWP_2 only): 1 feature * 1 source = 1 col
    - Total: 24 NWP feature columns (vs 9 in the original averaging approach)

    Args:
        station: Station number (1-10)
        keep_ensemble_stats: If True, also add ensemble mean/std/min/max as
                             additional features for each common variable.

    Returns:
        DataFrame with per-source NWP features, resampled to 15-min resolution.
    """
    train_path_template = "../dataset/NewEnergy/nwp_data_train/{}/{}/{}.nc"

    date_range = pd.date_range(start='2024-01-01', end='2024-12-30')
    date_range = [date.strftime('%Y%m%d') for date in date_range]

    print(f"Loading NWP data for station {station} (v2: per-source features)...")
    nwp_1_data = pd.concat([get_data(train_path_template.format(station, 'NWP_1', str(date))) for date in date_range], axis=0).reset_index(drop=True)
    nwp_2_data = pd.concat([get_data(train_path_template.format(station, 'NWP_2', str(date))) for date in date_range], axis=0).reset_index(drop=True)
    nwp_3_data = pd.concat([get_data(train_path_template.format(station, 'NWP_3', str(date))) for date in date_range], axis=0).reset_index(drop=True)

    # Dynamically discover feature overlap across NWP sources
    c_nwp_1_3_data = nwp_1_data.columns
    c_nwp_2_data = nwp_2_data.columns
    features_common = sorted(set(c_nwp_1_3_data) & set(c_nwp_2_data))
    features_1_3 = sorted(set(c_nwp_1_3_data) - set(features_common))
    features_2 = sorted(set(c_nwp_2_data) - set(features_common))

    print(f"  Common features (all 3 sources): {features_common}")
    print(f"  NWP_1 & NWP_3 only features: {features_1_3}")
    print(f"  NWP_2 only features: {features_2}")

    result_data = pd.DataFrame()

    # --- Common features: keep all three sources separately ---
    for feature in features_common:
        result_data[f'{feature}_nwp1'] = nwp_1_data[feature].values
        result_data[f'{feature}_nwp2'] = nwp_2_data[feature].values
        result_data[f'{feature}_nwp3'] = nwp_3_data[feature].values

        if keep_ensemble_stats:
            vals = np.stack([nwp_1_data[feature].values,
                             nwp_2_data[feature].values,
                             nwp_3_data[feature].values], axis=0)
            result_data[f'{feature}_mean'] = np.mean(vals, axis=0)
            result_data[f'{feature}_std']  = np.std(vals, axis=0)
            result_data[f'{feature}_min']  = np.min(vals, axis=0)
            result_data[f'{feature}_max']  = np.max(vals, axis=0)

    # --- Features only in NWP_1 and NWP_3 (sp): keep both ---
    for feature in features_1_3:
        result_data[f'{feature}_nwp1'] = nwp_1_data[feature].values
        result_data[f'{feature}_nwp3'] = nwp_3_data[feature].values

        if keep_ensemble_stats:
            vals = np.stack([nwp_1_data[feature].values,
                             nwp_3_data[feature].values], axis=0)
            result_data[f'{feature}_mean'] = np.mean(vals, axis=0)
            result_data[f'{feature}_std']  = np.std(vals, axis=0)

    # --- Features only in NWP_2 (msl): keep as single source ---
    for feature in features_2:
        result_data[f'{feature}_nwp2'] = nwp_2_data[feature].values
        if keep_ensemble_stats:
            result_data[f'{feature}_mean'] = nwp_2_data[feature].values

    # --- Resample from hourly to 15-min resolution ---
    start_time = pd.Timestamp('2024-01-02')
    weather_time_index = start_time + pd.to_timedelta(result_data.index, unit='h')
    result_data.index = weather_time_index
    result_data = result_data.resample('15min').interpolate()

    # Fill any NaN at edges that interpolation couldn't reach
    if result_data.isnull().any().any():
        n_edge = result_data.isnull().sum().sum()
        result_data = result_data.ffill().bfill()
        print(f"  Fixed {n_edge} edge NaN after resample+interpolate")

    # Pad the last 45 minutes of the year (3 missing 15-min steps)
    additional_data = pd.DataFrame(
        [result_data.iloc[-1]] * 3,
        index=pd.date_range(start='2024-12-31-23:15:00',
                            end='2024-12-31-23:45:00',
                            freq='15min')
    )
    result_data = pd.concat([result_data, additional_data], axis=0)

    print(f"  Output: {result_data.shape[1]} features, {result_data.shape[0]} timesteps")
    return result_data

    


def process_station(station, weather_fn, tag=''):
    """
    Merge weather data (from the given weather_fn) with power data for one station,
    reorder columns to [date, ...features..., OT], and save as CSV.
    """
    data_weather = weather_fn(station)
    data_power = pre_process_power_1(station)
    data_power.index = data_weather.index
    data = pd.concat([data_weather, data_power], axis=1)

    # Reorder: put date (second-to-last in power data) first, OT (last) last
    columns = list(data.columns)
    # '时间' is the date column from power data, '功率(MW)' is the target
    columns = [columns[-2]] + columns[:-2] + [columns[-1]]
    data = data[columns].rename(columns={'时间': 'date', '功率(MW)': 'OT'})

    # --- final validation ---
    # Check for any remaining NaN across all columns
    nan_cols = data.columns[data.isnull().any()].tolist()
    if nan_cols:
        nan_counts = {c: data[c].isnull().sum() for c in nan_cols}
        print(f"  WARNING: remaining NaN in columns: {nan_counts} — filling with ffill/bfill")
        data = data.ffill().bfill()

    # Check for dead (constant) columns
    for c in data.columns:
        if c != 'date' and data[c].nunique() <= 1:
            print(f"  WARNING: column '{c}' is constant — model may ignore it")

    # Check OT is in valid range [0, 1]
    ot_min, ot_max = data['OT'].min(), data['OT'].max()
    if ot_min < 0 or ot_max > 1.5:
        print(f"  WARNING: OT range [{ot_min:.4f}, {ot_max:.4f}] outside expected [0, 1]")

    prefix = 'wind' if station <= 5 else 'solar'
    output_path = f"../dataset/NewEnergy/{prefix}{station}{tag}.csv"
    data.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}  ({data.shape[1]} cols, {data.shape[0]} rows)")
    return data


if __name__ == '__main__':
    stations = np.arange(1, 11)

    # ====================================================================
    # Option 1: Original approach (averaged NWP sources)
    #   → 9 NWP features, use --enc_in 10 when training
    # ====================================================================
    # for station in stations[:5]:
    #     process_station(station, pre_process_weather_1)
    # for station in stations[5:]:
    #     process_station(station, pre_process_weather_1)

    # ====================================================================
    # Option 2: IMPROVED approach (per-source NWP features, no averaging)
    #   → 23 NWP features, use --enc_in 24 when training
    #   Columns: date, ghi_nwp1, ghi_nwp2, ghi_nwp3, v100_nwp1, ...,
    #            sp_nwp1, sp_nwp3, msl_nwp2, OT
    # ====================================================================
    for station in stations[:5]:   # wind stations 1-5
        process_station(station, pre_process_weather_v2, tag='_v2')
    for station in stations[5:]:   # solar stations 6-10
        process_station(station, pre_process_weather_v2, tag='_v2')

    # ====================================================================
    # Option 3: Per-source features + ensemble statistics
    #   → ~40+ NWP features, use --enc_in accordingly
    #   Uncomment below to use:
    # ====================================================================
    # for station in stations[:5]:
    #     process_station(station, lambda s: pre_process_weather_v2(s, keep_ensemble_stats=True),
    #                     tag='_v2_ens')
