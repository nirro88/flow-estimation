import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_file = r'C:\Users\nirro\Desktop\machine learning\ayyeka\models\model_4\data_site_EMCI_33414008.csv'
data_for_model = pd.read_csv(path_file)
sub_data_for_model = data_for_model.iloc[6000:,:].reset_index().drop(['index'],axis = 1)

list_of_level_ind = []
list_of_flow_ind =[]
for ind,measur in enumerate(sub_data_for_model.type_of_measurement):
    if sub_data_for_model.iloc[ind,4] == 'Level Calibration' and sub_data_for_model.iloc[ind+1,4] == 'Level Calibration':
        list_of_level_ind.append(ind)
    if sub_data_for_model.iloc[ind,4] == 'Level Calibration' and sub_data_for_model.iloc[ind+1,4] == 'Velocity' and sub_data_for_model.iloc[ind+2,4] != 'Flow':
        list_of_level_ind.append(ind)
        list_of_level_ind.append(ind+1)
    if sub_data_for_model.iloc[ind, 4] == 'Level Calibration' and sub_data_for_model.iloc[ind + 1, 4] == 'Flow' :
        list_of_level_ind.append(ind)
        list_of_level_ind.append(ind + 1)

sub_data_for_model = sub_data_for_model.drop(list_of_level_ind,axis=0).reset_index().drop(['index'],axis = 1)

flow_data = sub_data_for_model.loc[sub_data_for_model.type_of_measurement == 'Flow', :'type_of_measurement']
flow_series = flow_data['value'].values
for ind,value in enumerate(flow_series):
    if value <= 0:
        mask = flow_series[ind-5:ind+5].copy()
        mask[mask <= 0] = np.nan
        new_value = np.nanmedian(mask)
        flow_series[ind] = new_value

level_data = sub_data_for_model.loc[sub_data_for_model.type_of_measurement == 'Level Calibration', :'type_of_measurement']
level_series = level_data['value'].values
# #
velocity_data = sub_data_for_model.loc[sub_data_for_model.type_of_measurement == 'Velocity', :'type_of_measurement']
velocity_series = velocity_data['value'].values
for ind,value in enumerate(velocity_series):
    if value <= 0:
        mask = velocity_series[ind-5:ind+5].copy()
        mask[mask <= 0] = np.nan
        new_value = np.nanmedian(mask)
        velocity_series[ind] = new_value

x = flow_data.iloc[:,2].values
y1 = flow_series[:]
y2 = level_series[:]
y3 = velocity_series[:]

plt.plot(x, y1, "-b", label="flow")
plt.plot(x, y2, "-r", label="level")
plt.plot(x, y3, "-g", label="velocity")
plt.legend(loc="upper left")
plt.ylim(-10, 50)
plt.xticks(x[1::1000],rotation=90)
plt.show()

a= np.vstack([y1,y2,y3]).T
sub_data_for_model = pd.DataFrame(data=a,index=x,columns=['flow','level','velocity'])
# sub_data_for_model.to_csv(r'C:\Users\nirro\Desktop\machine learning\ayyeka\models\model_4\sub_data_for_model_4.csv')
