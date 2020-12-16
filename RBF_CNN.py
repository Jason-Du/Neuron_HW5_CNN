from keras.layers import Layer
from keras import backend as K
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
import pandas as pd
import numpy as np
import os
from keras.layers import Dense, Flatten,Dropout,Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint
import h5py
def isGroup(obj):
    if isinstance(obj, h5py.Group):
        return True
    return False


def isDataset(obj):
    if isinstance(obj, h5py.Dataset):
        return True
    return False


def getdatastefromgroup(dataset, obj):
    if (isGroup(obj)):
        for key in obj:
            x = obj[key]
            getdatastefromgroup(dataset, x)
    else:
        dataset.append(obj)


def geWeightforlayer(layername, filename):
    weight = []
    with h5py.File(filename, mode='r') as f:
        for keys in f:
            if layername in keys:
                obj = f[keys]
                datasets = []
                getdatastefromgroup(datasets, obj)
            for dataset in datasets:
                w = np.array(dataset)
                weight.append(w)
    return weight


def printWeightforlayer(layername,filename):
    with h5py.File(filename,mode='r') as f:
        for keys in f:
            print(keys,f[keys])
            o=f[keys]
            for key1 in o:
                print(key1,o[key1])
                r=o[key1]
                for key2 in r:
                    print(key2,r[key2])
def read_data(data_dirname_path):
    Trainx = np.load(os.path.join(data_dirname_path,"trainx.npy"))
    Trainy = np.load(os.path.join(data_dirname_path,"trainy.npy"))
    # Vaildx = np.load(os.path.join(data_dirname_path,"Vaildx.npy"))
    # Vaildy = np.load(os.path.join(data_dirname_path,"Vaildy.npy"))
    Testx  = np.load(os.path.join(data_dirname_path,"testx.npy"))
    Testy  = np.load(os.path.join(data_dirname_path,"testy.npy"))


    Trainx =Trainx.reshape(-1,28,28,1)
    # Vaildx = Vaildx.reshape(-1,166, 586, 1)
    Testx  = Testx.reshape(-1, 28,28, 1)


    Trainy  =np_utils.to_categorical(Trainy,num_classes=10)
    # Vaildy  =np_utils.to_categorical(Vaildy,num_classes=12)
    Testy   =np_utils.to_categorical(Testy,num_classes=10)
    return Trainx, Trainy,Testx, Testy
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        # res = 1/K.sqrt(1 + self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def RBF():
    model = Sequential()
    model.add(Dense(128, input_dim=(28*28), activation='relu'))
    # model.add(Conv2D(filters=6,input_dim=(28*28),activation='sigmoid',padding="valid",strides=(1, 1),kernel_size=(5,5)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(RBFLayer(64, 0.5))
    # model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    # 保存模型
    model.save('RBF.h5')
    return model
def NN():

    model = Sequential()
    # model.save_weights(checkpoint_path.format(epoch=0))
    model.add(Dense(10, input_dim=(28*28), activation='softmax'))

    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(10, activation='softmax'))
    # 查看网络结构
    model.summary()
    # 保存模型
    return model
def RBF_CNN():
    model = Sequential()
    model.add(Conv2D(filters=6,input_shape=(28,28,1),activation='relu',padding="valid",strides=(1, 1),kernel_size=(5,5),data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid"))
    # model.add(Conv2D(filters=12,activation='relu',padding="valid",strides=(1, 1),kernel_size=(5,5),data_format='channels_last'))
    # model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid"))
    model.add(Flatten())
    model.add(RBFLayer(128, 0.5))
    # model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    # 保存模型
    model.save('RBF_CNN.h5')
    return model
def CNN():
    model = Sequential()
    model.add(Conv2D(filters=6,input_shape=(28,28,1),activation='relu',padding="valid",strides=(1, 1),kernel_size=(5,5),data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid"))
    model.add(Conv2D(filters=12,activation='relu',padding="valid",strides=(1, 1),kernel_size=(5,5),data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding="valid"))
    model.add(Flatten())
    model.add(Dense(10,activation='softmax'))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='softmax'))
    model.summary()
    # 保存模型
    model.save('CNN.h5')
    return model
if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    Trainx, Trainy,Testx, Testy=read_data(os.path.join(os.path.dirname(__file__),'mnist_dataset'))
    print(Trainy.shape)
    print(Trainx.shape)
    print(Testy.shape)
    print(Testx.shape)
    print(np.max(Trainx[1,15,:,0]))
    # os.system('pause')
    # model_name='NN_mnist_weight'
    # model_name='NN'
    model_name='RBF'
    if model_name=='RBF_CNN':
        # print(Trainx.shape)
        # os.system('pause')
        model = RBF_CNN()
        model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
        History = model.fit(Trainx, Trainy, batch_size=300, epochs=60, verbose=2,validation_data=(Testx,Testy))
        pre = model.evaluate(Testx, Testy, batch_size=300, verbose=2)
        print('test_loss:', pre[0], '- test_acc:', pre[1])
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(History.history['accuracy'])
        plt.plot(History.history['val_accuracy'])
        plt.title('RBF accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.subplot(1, 2, 2)
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('RBF loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.show()
    if model_name=='CNN':
        # Trainx = Trainx.reshape(Trainx.shape[0], (28*28))
        # Testx = Testx.reshape(Testx.shape[0], (28*28))
        # print(Trainx.shape)
        # os.system('pause')
        model = CNN()
        model.compile(optimizer=Adam(lr=0.00025, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
        History = model.fit(Trainx, Trainy, batch_size=300, epochs=40, verbose=2,validation_data=(Testx,Testy))
        pre = model.evaluate(Testx, Testy, batch_size=100, verbose=2)
        print('test_loss:', pre[0], '- test_acc:', pre[1])
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(History.history['accuracy'])
        plt.plot(History.history['val_accuracy'])
        plt.title('RBF accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.subplot(1, 2, 2)
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('RBF loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.show()
    if model_name=='RBF':
        Trainx = Trainx.reshape(Trainx.shape[0], (28*28))
        Testx = Testx.reshape(Testx.shape[0], (28*28))
        # print(Trainx.shape)
        # os.system('pause')
        model = RBF()
        model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
        History = model.fit(Trainx, Trainy, batch_size=300, epochs=40, verbose=2,validation_data=(Testx,Testy))
        pre = model.evaluate(Testx, Testy, batch_size=100, verbose=2)
        print('test_loss:', pre[0], '- test_acc:', pre[1])
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(History.history['accuracy'])
        plt.plot(History.history['val_accuracy'])
        plt.title('RBF accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.subplot(1, 2, 2)
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('RBF loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.show()
    if model_name=='NN':
        checkpoint_path = "mnist_model/cp_{epoch:04d}.hdf5"

        checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                                   verbose=0,
                                   save_best_only=False,
                                   save_weights_only=True,
                                   mode="auto",
                                   period=1)
        Trainx = Trainx.reshape(Trainx.shape[0], (28 * 28))
        Testx = Testx.reshape(Testx.shape[0], (28 * 28))
        # print(Trainx.shape)
        # os.system('pause')
        model = NN()
        model.compile(optimizer=Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
        History = model.fit(Trainx, Trainy, batch_size=300, epochs=20, verbose=2,validation_data=(Testx,Testy),callbacks=[checkpoint])
        pre = model.evaluate(Testx, Testy, batch_size=100, verbose=2)
        print('test_loss:', pre[0], '- test_acc:', pre[1])

        from tensorflow.python import pywrap_tensorflow
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(History.history['accuracy'])
        plt.plot(History.history['val_accuracy'])
        plt.title('NN accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.subplot(1, 2, 2)
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('NN loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.show()
    if model_name=='NN_mnist_weight':


        path=os.path.join(os.path.dirname(__file__),'mnist_model')

        allFileList = os.listdir(path)
        # printWeightforlayer(layername='',filename=os.path.join(path,'cp_0001.hdf5'))
        epoch_weight=[]
        epoch_bias=[]
        for file in allFileList:
            weight = geWeightforlayer(layername="dense_1", filename=os.path.join(path,file))
            layerweight=weight[1].flatten()
            layerbias=weight[0]
            epoch_weight.append(layerweight)
            epoch_bias.append(layerbias)
        epoch_weight=np.array(epoch_weight)
        epoch_bias=np.array(epoch_bias)

        plt.figure(figsize=(7, 5))
        # 設定圖的範圍, 不設的話，系統會自行決定
        plt.xlim(1, 10)
        plt.ylim(-0.25,0.25)
        # 照需要寫入x 軸和y軸的 label 以及title
        plt.xlabel("epoch")
        plt.ylabel("weight_value")
        plt.title("The Title")
        x=[i+1 for i in range(epoch_weight.shape[0])]
        for line_num in range(epoch_weight.shape[1]):
            single_weight=epoch_weight[:,line_num]
            print(single_weight)
            plt.plot(x,single_weight)

            # os.system('pause')
        plt.show()