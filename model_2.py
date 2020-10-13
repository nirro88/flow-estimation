import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_file = r'C:\Users\nirro\Desktop\machine learning\ayyeka\models\model_2\data_site_AMCI_29073025.csv'
data_for_model = pd.read_csv(path_file)
# sub_data_for_model = data_for_model.iloc[:134002,:]

flow_data = data_for_model.loc[data_for_model.type_of_measurement == 'Flow', :'type_of_measurement']
flow_series = flow_data['value'].values
# for ind,value in enumerate(flow_series):
#     if value <= 0:
#         mask = flow_series[ind-5:ind+5].copy()
#         mask[mask <= 0] = np.nan
#         new_value = np.nanmedian(mask)
#         flow_series[ind] = new_value

level_data = data_for_model.loc[data_for_model.type_of_measurement == 'Level Calibration', :'type_of_measurement']
level_series = level_data['value'].values
# #
velocity_data = data_for_model.loc[data_for_model.type_of_measurement == 'Velocity', :'type_of_measurement']
velocity_series = velocity_data['value'].values
# for ind,value in enumerate(velocity_series):
#     if value <= 0:
#         mask = velocity_series[ind-5:ind+5].copy()
#         mask[mask <= 0] = np.nan
#         new_value = np.nanmedian(mask)
#         velocity_series[ind] = new_value

x = flow_data.iloc[:len(flow_series),2].values
y1 = flow_series[:len(flow_series)]
y2 = level_series[:len(flow_series)]
y3 = velocity_series[:len(flow_series)]

plt.plot(x, y1, "-b", label="flow")
plt.plot(x, y2, "-r", label="level")
plt.plot(x, y3, "-g", label="velocity")
plt.legend(loc="upper left")
plt.ylim(-1.5, 300)
plt.xticks(x[1::60],rotation=90)
plt.show()

# a= np.vstack([y1,y2,y3]).T
# sub_data_for_model = pd.DataFrame(data=a,index=x)
# sub_data_for_model.to_csv(r'C:\Users\nirro\Desktop\machine learning\ayyeka\models\model_2\sub_data_for_model.csv')
