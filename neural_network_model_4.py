import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import tensorflow as tf
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout, LeakyReLU, PReLU
from tensorflow.keras.models import Model, Sequential
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
import pydot
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
import kerastuner as kt
from kerastuner import HyperModel, RandomSearch
import os
from tensorflow.keras.models import load_model


class PredictFlow():
    def __init__(self, data):
        self.data = data

    def pre_processing(self):
        scaler_flow = MinMaxScaler(feature_range=(0, 1))
        scaler_level = MinMaxScaler(feature_range=(0, 1))
        scaler_velocity = MinMaxScaler(feature_range=(0, 1))

        primer_flow = np.asarray(self.data.iloc[:, 1]).reshape(-1, 1)
        primer_level = np.asarray(self.data.iloc[:, 2]).reshape(-1, 1)
        primer_velocity = np.asarray(self.data.iloc[:, 3]).reshape(-1, 1)
        primer_time = np.asarray(self.data.iloc[:, 0]).reshape(-1, 1)

        log_flow = np.log(primer_flow)
        log_level = np.log(primer_level)
        log_velocity = np.log(primer_velocity)

        scaled_flow = scaler_flow.fit_transform(log_flow)
        scaled_level = scaler_level.fit_transform(log_level)
        scaled_velocity = scaler_velocity.fit_transform(log_velocity)

        a = np.hstack([primer_time, scaled_flow, scaled_level, scaled_velocity])
        new_data = pd.DataFrame(data=a, columns=['time', 'flow', 'level', 'velocity'])
        return new_data, scaler_flow, scaler_level, scaler_velocity

    def split_data(self, data):
        train_size = int(len(data) * 0.33)
        valid_size = int(len(data) * 0.67)
        # split into 3 sub data sets: train validation and test sets
        data_array = []
        for i in range(4):
            data_array.append(data.iloc[:train_size, i])
        for i in range(4):
            data_array.append(data.iloc[train_size:valid_size, i])
        for i in range(4):
            data_array.append(data.iloc[valid_size:, i])
        return data_array[0], data_array[1], data_array[2], data_array[3], data_array[4], data_array[5], data_array[6], data_array[7], data_array[8], data_array[9], data_array[10], data_array[11]

    def organized_first_prediction_data(self, level, velocity):
        x_model1, y_model1 = [], []
        for i in range(6, len(level)):
            x_model1.append(level[i-6:i])
            y_model1.append(velocity.iloc[i-1])
        x_model1 = np.asarray(x_model1)
        y_model1 = np.asarray(y_model1)
        x_model1 = np.reshape(x_model1, (x_model1.shape[0], x_model1.shape[1], 1))
        return x_model1, y_model1

    def plot(self, time, predict_data, true_data):
        # plot the predicted flow with the true flow measured
        figure = plt.figure()
        plt.plot(time, predict_data, "-b", label="predict")
        plt.plot(time, true_data, "-r", label="True")
        plt.legend(loc="upper left")
        plt.ylim(-1.5, 40)
        plt.xticks(time[1::500], rotation=90)
        figure.show()

    def plot_data(self, time, level_data, flow_data):
        # plot the level with the flow measured
        plt.plot(time, level_data, "-b", label="level_data")
        plt.plot(time, flow_data, "-r", label="flow_data")
        plt.legend(loc="upper left")
        plt.ylim(-1.5, 60)
        plt.xticks(time[1::500], rotation=90)
        plt.show()

    def prepare_data_model2(self, valid_level, predict_velocity_model1, valid_flow):
        # prepare the data for the next model: prediction of the flow
        level_input, predict_velocity, flow_y_model2 = [], [], []
        for i in range(12, len(valid_level)):
            level_input.append(valid_level.iloc[i-12:i])
            predict_velocity.append(predict_velocity_model1[i-12])
            flow_y_model2.append(valid_flow.iloc[i-1])
        level_input = np.asarray(level_input)
        predict_velocity = np.asarray(predict_velocity)
        flow_y_model2 = np.asarray(flow_y_model2)

        level_input = np.reshape(level_input, (level_input.shape[0], level_input.shape[1], 1))
        predict_velocity = np.reshape(predict_velocity, (predict_velocity.shape[0], 1, 1))
        return level_input, predict_velocity, flow_y_model2

    def simple_net(self, level_input, flow_output):
        input = Input(shape=(1, ))
        hidden1 = Dense(1, activation='tanh', kernel_initializer='glorot_uniform')(input)
        # drop1 = Dropout(0.1)(hidden1)
        hidden2 = Dense(8, activation='relu')(hidden1)
        drop2 = Dropout(0.4)(hidden2)
        hidden3 = Dense(4, activation='relu')(drop2)
        drop3 = Dropout(0.2)(hidden3)
        output = Dense(1, kernel_regularizer=regularizers.l2(l=0.001), bias_regularizer=regularizers.l2(l=0.1))(drop3)
        model = Model(inputs=input, outputs=output)

        # # Compile and simple net
        lr = ReduceLROnPlateau()
        loss = tf.keras.losses.Huber(name='huber_loss', delta=0.1)
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        model.compile(optimizer='adam', loss=loss)
        model.fit(level_input, flow_output, epochs=50, batch_size=50, callbacks=[lr])
        return model

    def simple_lstm_net(self, level_input, flow_output):
        input = Input(shape=(6,1))
        hidden1 = LSTM(6, activation='tanh', return_sequences=True)(input)
        drop1 = Dropout(0.3)(hidden1)
        hidden2 = LSTM(2, activation='tanh')(drop1)
        output = Dense(1)(hidden2)
        model = Model(inputs=input, outputs=output)

        # # Compile and fit model2
        # lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        model.fit(level_input, flow_output, epochs=10, batch_size=10)
        return model

    def organized_simple_net_data(self, x_data, y_data):
        x_model, y_model = [], []
        for i in range(len(x_data)):
            x_model.append(x_data.iloc[i])
            y_model.append(y_data.iloc[i])
        x_model = np.asarray(x_model).reshape(-1, 1)
        y_model = np.asarray(y_model).reshape(-1, 1)
        return x_model, y_model

    def organized_simple_lstm_net_data(self, x_data, y_data):
        x_model, y_model = [], []
        for i in range(len(x_data)):
            x_model.append(x_data.iloc[i])
            y_model.append(y_data.iloc[i])
        x_model = np.asarray(x_model).reshape(-1,1)
        y_model = np.asarray(y_model).reshape(-1,1)
        x_model = np.reshape(x_model, (x_model.shape[0], 1, 1))
        return x_model, y_model
    
    def new_net(self, level_input, flow_output):
        input = Input(shape=(1, ))
        hidden1 = Dense(6, activation='selu', kernel_initializer='he_uniform', bias_initializer='random_uniform', activity_regularizer=regularizers.l2(0.01), use_bias=True)(input)
        # hidden1 = LeakyReLU(alpha=0.4)(hidden1)
        # drop1 = AlphaDropout(0.1)(hidden1)
        hidden2 = Dense(6, activation='selu', kernel_initializer='he_uniform', bias_initializer='random_uniform', use_bias=True)(hidden1)
        # hidden2 = LeakyReLU(alpha=0.4)(hidden2)
        drop2 = Dropout(0.3)(hidden2)
        output = Dense(1)(drop2)
        model = Model(inputs=input, outputs=output)

        # # Compile and simple net
        lr = ReduceLROnPlateau()
        # log_cosh_loss = tf.keras.losses.logcosh(y_true=flow_output,y_pred=output)
        # kullback_leibler_loss = tf.keras.losses.kullback_leibler_divergence(y_true,y_pred)
        huber_loss = tf.keras.losses.Huber(name='huber_loss', delta=0.1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=270, min_delta=1)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        history = model.fit(level_input, flow_output, validation_split=0.2, epochs=2000, batch_size=4000, callbacks=[lr,es])
        return model, history

    def tuned_net(self, level_input, flow_output):
        # run1 = {'units': 8, 'activation': 'relu', 'kernel_initializer': 'random_normal','bias_initializer': 'lecun_normal', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
        # run2 = {'units': 8, 'activation': 'elu', 'kernel_initializer': 'random_normal','bias_initializer': 'lecun_normal', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
        # run3 = {'units': 8, 'activation': 'elu', 'kernel_initializer': 'random_uniform','bias_initializer': 'random_normal', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
        # run4 = {'units': 6, 'activation': 'elu', 'kernel_initializer': 'random_normal','bias_initializer': 'random_uniform', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
        # run5 = {'units': 8, 'activation': 'elu', 'kernel_initializer': 'random_uniform','bias_initializer': 'lecun_normal', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
        input = Input(shape=(1, ))
        hidden1 = Dense(8, activation='elu', kernel_initializer='random_normal', bias_initializer='lecun_normal', activity_regularizer=regularizers.l2(0.001), use_bias=True)(input)
        # drop1 = AlphaDropout(0.1)(hidden1)
        hidden2 = Dense(8, activation='elu', kernel_initializer='random_normal', bias_initializer='lecun_normal', use_bias=True)(hidden1)
        # drop2 = Dropout(0.3)(hidden2)
        output = Dense(1)(hidden2)
        model = Model(inputs=input, outputs=output)

        # # Compile simple net
        lr = ReduceLROnPlateau()
        huber_loss = tf.keras.losses.Huber(name='huber_loss', delta=0.1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.000001,restore_best_weights=True)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse', metrics=['mse'])
        history = model.fit(level_input, flow_output, validation_split=0.2, epochs=1000, batch_size=100, callbacks=[lr, es, mc])
        return model, history, es

    def inverse_data(self, predict_array, true_array, scaler):
        predict_array = scaler.inverse_transform(predict_array)
        true_array = np.asarray(true_array).reshape(-1, 1)
        true_array = scaler.inverse_transform(true_array)

        predict_array = np.exp(predict_array)
        true_array = np.exp(true_array)
        return predict_array, true_array

class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        kernel_initializers_list = ['random_normal', 'random_uniform', 'lecun_normal', 'he_normal', 'he_uniform']
        activation_function_list = ['relu', 'selu', 'elu', LeakyReLU, PReLU]
        units_list = [2,4,6,8,10]
        bias_initializer_list = ['random_normal', 'random_uniform', 'lecun_normal', 'he_normal', 'he_uniform', 'zeros']
        activity_regularizer_list = [1e-1, 1e-2, 1e-3]
        dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        learning_rate_list = [0.1, 0.01, 0.001, 0.0001]

        model = Sequential()
        model.add(
            layers.Dense(
                units=hp.Int('units',min_value=6, max_value=8,step=2),
                activation=hp.Choice('activation', values=['relu', 'elu']),
                kernel_initializer=hp.Choice('kernel_initializer',values=['random_normal', 'random_uniform']),
                bias_initializer=hp.Choice('bias_initializer',values=['random_normal', 'random_uniform', 'lecun_normal']),
                activity_regularizer=regularizers.l2(hp.Choice('regularizers.l2',values=[1e-2, 1e-3])),
                use_bias=True,
                input_shape=self.input_shape))
        # model.add(layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.1, default=0.1, step=0.1)))
        model.add(
            layers.Dense(
                units=hp.Int('units',min_value=6, max_value=8,step=2),
                activation=hp.Choice('activation', values=['relu', 'elu']),
                kernel_initializer=hp.Choice('kernel_initializer',values=['random_normal', 'random_uniform']),
                bias_initializer=hp.Choice('bias_initializer',values=['random_normal', 'random_uniform', 'lecun_normal']),
                use_bias=True
                )
            )
        model.add(layers.Dense(1))

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])), loss='mse', metrics=['mse'])
        return model


if __name__ == '__main__':

    # -----------------------------Load in the data---------------------------------------------

    path_file = r'C:\Users\nirro\Desktop\machine learning\ayyeka\models\model_1\sub_data_for_model_1.csv'
    all_data = pd.read_csv(path_file)

    # take a sub set of the main data without many outliers
    sub_all_data = all_data.iloc[55000:, :]

    # create an object from class predict_flow
    obj1 = PredictFlow(sub_all_data)

    # ----------------------------pre-processing stage------------------------------------------

    scaled_data, scaler_flow, scaler_level, scaler_velocity = obj1.pre_processing()

    # split the new data after pre-processing to train validation and test data-sets
    train_time, train_flow, train_level, train_velocity, valid_time, valid_flow, valid_level, valid_velocity, test_time, test_flow, test_level, test_velocity = obj1.split_data(scaled_data)

    # organization the data in correct shape for the model
    x_train_simple_net, y_train_simple_net = obj1.organized_simple_net_data(train_level, train_flow)
    x_valid_simple_net, y_valid_simple_net = obj1.organized_simple_net_data(valid_level, valid_flow)
    x_test_simple_net, y_test_simple_net = obj1.organized_simple_net_data(test_level, test_flow)

    # -------------------------find hyper-parameters with keras tuner----------------------------

    log_dir = os.path.normpath(r'C:\Users\nirro\Desktop\machine learning\ayyeka\models\model_4')
    input_shape = x_train_simple_net.shape[1]
    hypermodel = RegressionHyperModel((input_shape,))

    tuner_rs = RandomSearch(hypermodel,objective='mse',seed=42,max_trials=30,executions_per_trial=2,directory=log_dir)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True,min_delta=0.00001,verbose=1)]
    tuner_rs.search(x_train_simple_net, y_train_simple_net, epochs=1000, validation_split=0.2, verbose=0, callbacks=callbacks)

    best_hp = tuner_rs.get_best_hyperparameters(num_trials=10)
    model = tuner_rs.hypermodel.build(best_hp)

    # run1 = {'units': 8, 'activation': 'relu', 'kernel_initializer': 'random_normal', 'bias_initializer': 'lecun_normal', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
    # run2 = {'units': 8, 'activation': 'elu', 'kernel_initializer': 'random_normal', 'bias_initializer': 'lecun_normal', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
    # run3 = {'units': 8, 'activation': 'elu', 'kernel_initializer': 'random_uniform', 'bias_initializer': 'random_normal', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
    # run4 = {'units': 6, 'activation': 'elu', 'kernel_initializer': 'random_normal', 'bias_initializer': 'random_uniform', 'regularizers.l2': 0.001, 'learning_rate': 0.001}
    # run5 = {'units': 8, 'activation': 'elu', 'kernel_initializer': 'random_uniform', 'bias_initializer': 'lecun_normal', 'regularizers.l2': 0.001, 'learning_rate': 0.001}

    best_model = tuner_rs.get_best_models(num_models=1)[0]
    # loss, mse = best_model.evaluate(x_valid_simple_net, y_valid_simple_net)

    # ----------------------------after finding the best hyper-parameters train the model-----------------------------

    # net_model, model_history, early_stoping = obj1.tuned_net(x_train_simple_net, y_train_simple_net)

    # extract the best model in the training and load
    # saved_model = load_model(r'C:\Users\nirro\Desktop\machine learning\ayyeka\models\model_4\best_model.h5')

    # weigths = early_stoping.best_weights

    # -----------------------------------------make prediction--------------------------------------------------------

    # test_predicted_flow = saved_model.predict(x_test_simple_net)
    # valid_predicted_flow = saved_model.predict(x_valid_simple_net)

    # return the to the original values
    # test_predicted_flow, test_true_flow = obj1.inverse_data(test_predicted_flow, test_flow,scaler_flow)
    # valid_predicted_flow, valid_true_flow = obj1.inverse_data(valid_predicted_flow, valid_flow, scaler_flow)

    # b = np.hstack([test_predicted_flow, test_true_flow])
    # evaluation_data = pd.DataFrame(data=b, columns=['pred', 'true'],index=test_time)
    # evaluation_data.to_excel(r'C:\Users\nirro\Desktop\machine learning\ayyeka\models\model_4\evaluation_data2.xlsx')

    # ----------------------------------------------plot the results---------------------------------------------------

    # show validation data prediction vs true values of flow
    # obj1.plot(valid_time, valid_predicted_flow, valid_true_flow)

    # show test data prediction vs true values of flow
    # obj1.plot(test_time, test_predicted_flow, test_true_flow)

    # summarize history for loss
    # figure_2 = plt.figure()
    # plt.plot(model_history.history['loss'])
    # plt.plot(model_history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'valid'], loc='upper left')
    # figure_2.show()
