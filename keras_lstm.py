import numpy as np

# Load the IMDb Dataset
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Reproducability over multiple runs lol
np.random.seed(7)

# Only considering the top 5000 words
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Truncate and pad [create batches of] input input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Define the network
model = Sequential()
# First layer: Embedded layer that uses a 32 length vector to
# represent each word. Embedded layers are denser than one-hot vectors
embedding_vector_length = 32
model.add(Embedding(top_words,
                    embedding_vector_length, input_length=max_review_length))
# The next layer is a LSTM layer with 100 neurons
model.add(LSTM(100))
# This is a classification problem, requiring the signle output to be 0 or 1.
# Sigmoid activation for the final layer
model.add(Dense(1, activation='sigmoid'))
# Loss is cross entropy
# Optimizer is Adam
# We score it on accuracy
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# How good is our model?
print(model.summary())
# Real world tests
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3,
          batch_size=64)
# Final scores
scores = model.evaluate(X_test, y_test, verbose=0)
print ("Accuracy: %.2f%%" % (scores[1] * 100))
