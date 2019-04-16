##################### Kaggle FASHION MNIST #################

## Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout

## importing data
data = pd.read_csv('fashion-mnist_train.csv')
data = data.values  ## converts the dataframe to numpy
np.random.shuffle(data)
test_data = pd.read_csv('fashion-mnist_test.csv')
test_data = test_data.values

## helper function
def yconverter(y):
    n=len(y)
    k=len(set(y))
    I=np.zeros((n,k))
    I[np.arange(n),y]=1
    return I

x = data[:, 1:].reshape(-1,28,28,1)/255.0  ## we want data in the form of N/H/W/Color
y = data[:, 0].astype(np.int32)
k = len(set(y))
y = yconverter(y)

## test data
x_test = test_data[:,1:].reshape(-1,28,28,1)/255.0
y_test = test_data[:,0].astype(np.int32)
y_test = yconverter(y_test)


### Making the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

## Adding other convolutional layers
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=k))
model.add(Activation('softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

r = model.fit(x, y, batch_size=256, epochs=10, validation_split=0.2)
print("Returned:", r)
print(r.history)

## testing on test data set
model.evaluate(x_test, y_test)

## plotting
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()












