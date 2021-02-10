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

# Adding the first and only layer
# Softmax is the sigmoid of categories in simple words. Instead of a binary vector
# it can be used with categories activating for each of them
# This layer takes the input layer, creates a layer of 10 and gets the probability
# of being in a category
# In summary 784 - 10 ANN, each of the 10 kinda sigmoid
model.add(keras.layers.Dense(classes,
                             input_shape=(flat_size,),
                             activation='softmax'))

# Compile, which just means, adding the super params
# SGD = Stochastic Gradient Descent (every batch the weights are updated based
# on this algorithm)
# loss function or cost function, it is just how far we are from the "perfect"
# outcome (we do not really want to be perfect though, we don't want to 
# overfit) Normally 80% accuracy is good
# accuracy, how much we hit right per category
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

# In my case, accuracy is around 91% with loss of 29%
# One thing is categorizing well, and another thing is how perfect is the 
# categorization, in this case loss measures the perfection

