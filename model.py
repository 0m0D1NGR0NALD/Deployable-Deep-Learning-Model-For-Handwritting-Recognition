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
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# Data preparation
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],1,rows,columns)
    x_test = x_test.reshape(x_test.shape[0],1,rows,columns)
    input_shape = (1,rows,columns)
else:
    x_train = x_train.reshape(x_train.shape[0],rows,columns,1)
    x_test = x_test.reshape(x_test.shape[0],rows,columns,1)
    input_shape = (rows,columns,1)

# Convert train and test set to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Scale train and test set 
x_train /= 255
x_test /= 255
# Convert class labels to binary class matrix
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

# Creating the Model
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Training the Model
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

# Evaluating the predictions on the Model
score = model.evaluate(x_test,y_test,verbose=0)
print('Test Loss: ',score[0])
print('Test Accuracy: ',score[1])

# Saving the model for future inferences
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
# Serialize weihgts to HDF5
model.save_weights("model.h5")