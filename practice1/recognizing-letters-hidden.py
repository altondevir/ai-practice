# First practice: recognizing letterss
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

epochs = 150    
batch_size = 200
classes = 10
hidden_layers = 128
split_rate = 0.2

# Digits classification dataset https://keras.io/api/datasets/mnist/
dataset = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = dataset.load_data()

# matrix if represented on a unidimensional vector will train the same
flat_size = X_train.shape[1]*X_train.shape[2]

# Reshaping so each row contains one pixel per neuron (pixels are black and white 0-255)
X_train = X_train.reshape(X_train.shape[0], flat_size)
X_test = X_test.reshape(X_test.shape[0], flat_size)

# Fit helps to identify the mean of the distribution and transform does the scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Doing the One hot representation of the labels, there are 10 digits, thus 10 classes
y_train = tf.keras.utils.to_categorical(y_train, classes)
y_test =  tf.keras.utils.to_categorical(y_test, classes)

# Creating the model
model = tf.keras.models.Sequential()

# This time I will add hidden layers to see if it improves and how much
model.add(keras.layers.Dense(hidden_layers,
                             input_shape=(flat_size,)))
model.add(keras.layers.Dense(hidden_layers,
                             activation='relu'))
model.add(keras.layers.Dense(hidden_layers,
                             activation='relu'))

# output layer
model.add(keras.layers.Dense(classes,
                             activation='softmax'))

# Compile
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy']) 

# Training the model
model.fit(X_train, y_train, batch_size=batch_size, 
          epochs=epochs, validation_split=split_rate)   

# Evaluate the modelwith new entry
loss, accuracy = model.evaluate(X_test,y_test)
print ("Accuracy: ", accuracy)
print ("Loss: ", loss)

# For this with three extra hidden layers we get better results so
# we can say that our model is going in the right direction without overfitting
# I get 97% of accuracy and 0.086 of loss which are very good numbers

