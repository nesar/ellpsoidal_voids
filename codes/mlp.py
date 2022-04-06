from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential
from keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Activation, Dropout, Flatten, Input
import tensorflow as tf
tf.random.set_seed(3)
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


def load_data(fileIn):
    
    allData = np.loadtxt(fileIn, skiprows=1)
    # all_df = pd.DataFrame(allData)
    # all_df.columns = allLabels
    
    cond = np.where(allData[:, -2] > 0)
    allData = allData[cond]
    # allData[:, -2] = np.log10(allData[:, -2])

    X = allData[:, :4]
    y = allData[:, 4:]
    return X, y


 

# load the dataset
fileIn = '../data/DOE1.out'
X, y = load_data(fileIn)

# encode strings to integer
#scaler = StandardScaler(with_mean=False)
scaler = StandardScaler()
#scaler = MinMaxScaler( feature_range=(y.min(), y.max()) )
y = scaler.fit_transform(y)

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# determine the number of input features
n_features = X_train.shape[1]
print(n_features)


p_dropout = 0.02
# define model

model = Sequential()
model.add(Dense(16, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(p_dropout))
model.add(Dense(y.shape[1], activation='linear'))


# inputs = Input(((n_features)))
# x = Dense(16, activation="relu", kernel_initializer='he_normal')(inputs)
# x = Dense(32, activation="relu", kernel_initializer='he_normal')(x)
# x = Dense(128, activation="relu", kernel_initializer='he_normal')(x)
# x = Dropout(p_dropout)(x)
# predictions = Dense(y.shape[1], activation='linear')(x)
# model = Model(inputs=inputs, outputs=predictions)



# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

ifTrain = True

if ifTrain: 
    # fit the model
    model.fit(X_train, y_train, epochs=500, batch_size=5, verbose=0)
    # evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)
    # save the model
    tf.keras.models.save_model(model, '../model/mlp43', overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)

    
    
    
########################
########################
# load a trained model
model = tf.keras.models.load_model('../model/mlp43')

# make a prediction
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

########################



f, a = plt.subplots(1, 2, figsize=(11, 5))
allLabels = ['rx', 'ry', 'rz', 'dist', 'Smax', 'Vol of high stress']

a[0].scatter(y_test[:, 0], y_pred[:, 0])
a[0].title.set_text(allLabels[4])
# a[0].set_xlim(0.2*np.min(y_test[:, 0]), 1.5*np.max(y_test[:, 0]))
# a[0].set_ylim(0.2*np.min(y_test[:, 0]), 1.5*np.max(y_pred[:, 0]))
a[0].set_xlabel('true', fontsize=20)
a[0].set_ylabel('pred', fontsize=20)
a[0].plot([0.9*np.min(y_test[:, 0]), 1.2*np.max(y_test[:, 0])], [0.9*np.min(y_test[:, 0]), 1.2*np.max(y_test[:, 0])], 'r')


a[1].scatter(y_test[:, 1], y_pred[:, 1])
a[1].title.set_text(allLabels[5])
# a[1].set_xlim(0.2*np.min(y_test[:, 1]), 1.5*np.max(y_test[:, 1]))
# a[1].set_ylim(0.2*np.min(y_test[:, 1]), 1.5*np.max(y_pred[:, 1]))
a[1].set_xlabel('true', fontsize=20)
a[1].set_ylabel('pred', fontsize=20)
a[1].plot([0,3e-10], [0,3e-10], 'r')

plt.savefig('../plots/mlp')
plt.show()





from tensorflow.python.keras.backend import eager_learning_phase_scope
from tensorflow.keras import backend as K

mc_samples = 100
f = K.function([model.layers[0].input], [model.output])
    
# Run the function for the number of mc_samples with learning_phase enabled
with eager_learning_phase_scope(value=1): # 0=test, 1=train
    Yt_hat = np.array([f((X_test))[0] for _ in range(mc_samples)])
    
y_mean = np.mean(Yt_hat, axis=0)
y_std = np.std(Yt_hat, axis=0)

y_mean = scaler.inverse_transform(y_mean)
y_std = scaler.inverse_transform(y_std)


f, a = plt.subplots(1, 2, figsize=(11, 5))
allLabels = ['rx', 'ry', 'rz', 'dist', 'Smax', 'Vol of high stress']

a[0].errorbar(y_test[:, 0], y_mean[:, 0], yerr = y_std[:, 0], fmt="o")
a[0].scatter(y_test[:, 0], y_pred[:, 0], marker='x', color='r')
a[0].title.set_text(allLabels[4])
# a[0].set_xlim(0.2*np.min(y_test[:, 0]), 1.5*np.max(y_test[:, 0]))
# a[0].set_ylim(0.2*np.min(y_test[:, 0]), 1.5*np.max(y_pred[:, 0]))
a[0].set_xlabel('true', fontsize=20)
a[0].set_ylabel('pred', fontsize=20)
a[0].plot([0.9*np.min(y_test[:, 0]), 1.2*np.max(y_test[:, 0])], [0.9*np.min(y_test[:, 0]), 1.2*np.max(y_test[:, 0])], 'r')


a[1].errorbar(y_test[:, 1], y_mean[:, 1], yerr = y_std[:, 1], fmt="o")
a[1].scatter(y_test[:, 1], y_pred[:, 1], marker='x')
a[1].title.set_text(allLabels[5])
# a[1].set_xlim(0.2*np.min(y_test[:, 1]), 1.5*np.max(y_test[:, 1]))
# a[1].set_ylim(0.2*np.min(y_test[:, 1]), 1.5*np.max(y_pred[:, 1]))
a[1].set_xlabel('true', fontsize=20)
a[1].set_ylabel('pred', fontsize=20)
a[1].plot([0,3e-10], [0,3e-10], 'r')

plt.savefig('../plots/mlp_drop')
plt.show()

