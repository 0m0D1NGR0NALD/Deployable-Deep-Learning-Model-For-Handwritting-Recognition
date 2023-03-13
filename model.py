import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import backend as K

# Defining hyperparameters
num_classes = 10
batch_size = 128
epochs = 10

# Image resolution
rows,columns = 28,28
# Loading the data
(x_train,y_train),(x_train,y_test) = mnist.load_data()