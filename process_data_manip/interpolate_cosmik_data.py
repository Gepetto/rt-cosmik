import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

data = pd.read_csv('q/q_cosmik_qp.csv')
data['time'] = pd.to_datetime(data['time'])
data['timestamp_seconds'] = (data['time'] - data['time'].iloc[0]).dt.total_seconds()
data['dt'] = data['timestamp_seconds']


dt = data['timestamp_seconds'][269]-data['timestamp_seconds'][268]
print(dt)
# for i in range (1,len(data['timestamp_seconds'])):
#     data['dt'][i-1] =  data['timestamp_seconds'][i] - data['timestamp_seconds'][i-1]


# for i in range (269,len(data['timestamp_seconds'])):
#     data['timestamp_seconds'][i] = data['timestamp_seconds'][i]- dt


plt.figure()
plt.plot(data['timestamp_seconds'])
plt.show()

start_time = data['timestamp_seconds'].iloc[0]
end_time = data['timestamp_seconds'].iloc[-1]
target_timestamps = np.arange(start_time, end_time, 1/33)   #0.05 ms = 20hz

# Create a new DataFrame for interpolated data
interpolated_data = pd.DataFrame({'timestamp_seconds': target_timestamps})


# Interpolate each column
for col in data.columns[1:]:
    print(col)
    interp_func = interp1d(data['timestamp_seconds'], data[col], kind='linear')
    interpolated_data[col] = interp_func(target_timestamps)
        # print(interpolated_data[col])


# Add interpolated time column
# start_datetime = data['time'].iloc[0]
# interpolated_data['time'] = pd.to_datetime(interpolated_data['timestamp_seconds'], unit='s', origin=start_datetime)

interpolated_data.to_csv('q/q_cosmik_qp_interpolated_33Hz.csv', index=False)
