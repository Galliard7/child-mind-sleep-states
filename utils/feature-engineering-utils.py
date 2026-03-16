def free_memory():
    
    import gc
    import ctypes

    libc = ctypes.CDLL("libc.so.6")
    
    _ = gc.collect()
    libc.malloc_trim(0)
#     torch.cuda.empty_cache()



def feature_engineering(data):
    
    import cudf
    import pandas as pd
    import numpy as np
    
    # Time-based Features
    data['timestamp'] = cudf.to_datetime(data['timestamp'])
    data['weekday'] = data['timestamp'].dt.weekday
    data['is_weekend'] = (data['weekday'] >= 5).astype(int)
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['day_of_month'] = data['timestamp'].dt.day

    # The step number per 24h period can be derived from the step column assuming each step is 5 seconds apart
    data['step_24h'] = (data['step'] % (24*60*12))

    # Statistical Features
    data['enmo_mean'] = data['enmo'].mean()
    data['enmo_var'] = data['enmo'].var()
    data['anglez_mean'] = data['anglez'].mean()
    data['anglez_var'] = data['anglez'].var()

    # Rolling Aggregates
    window_sizes = [int(x*12) for x in [5/60, 1, 2, 3, 4, 5, 6, 7, 8]]  # Convert hours to steps (5 seconds per step)
    for window in window_sizes:
        data[f'enmo_rolling_mean_{window}'] = data['enmo'].rolling(window=window).mean()
        data[f'enmo_rolling_max_{window}'] = data['enmo'].rolling(window=window).max()
        data[f'enmo_rolling_std_{window}'] = data['enmo'].rolling(window=window).std()

        data[f'anglez_rolling_mean_{window}'] = data['anglez'].rolling(window=window).mean()
        data[f'anglez_rolling_max_{window}'] = data['anglez'].rolling(window=window).max()
        data[f'anglez_rolling_std_{window}'] = data['anglez'].rolling(window=window).std()

        # For total variation (or first variation)
        data[f'enmo_1v_rolling_mean_{window}'] = data['enmo'].diff().rolling(window=window).mean()
        data[f'enmo_1v_rolling_max_{window}'] = data['enmo'].diff().rolling(window=window).max()
        data[f'enmo_1v_rolling_std_{window}'] = data['enmo'].diff().rolling(window=window).std()

        data[f'anglez_1v_rolling_mean_{window}'] = data['anglez'].diff().rolling(window=window).mean()
        data[f'anglez_1v_rolling_max_{window}'] = data['anglez'].diff().rolling(window=window).max()
        data[f'anglez_1v_rolling_std_{window}'] = data['anglez'].diff().rolling(window=window).std()
    
    return data