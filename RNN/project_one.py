import numpy as np
from tensorflow.keras.datasets import imdb
max_features = 1000
imdb.load_data(num_words=max_features)

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Training data shape: {len(X_train)}, Training labels shape: {len(y_train)}")
print(f"Test data shape: {len(X_test)}, Test labels shape: {len(y_test)}")

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

spilt_index = int(0.8 * len(X))

X_train = X[:spilt_index]
y_train = y[:spilt_index]

X_test = X[spilt_index:]
y_test = y[spilt_index:]

print("Training sample:", len(X_train))
print("Testing sample:", len(X_test))

print(y_train[0])
print(y_train[0])

from tensorflow.keras.preprocessing import sequence
maxlen = 200
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=32))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[earlystopping])

import matplotlib.pyplot as plt
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, color='blue', marker='o', label='Training accuracy')
plt.plot(epochs, val_accuracy, color='orange', marker='s', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(epochs, loss, color='red', marker='o', label='Training Loss')
plt.plot(epochs, val_loss, color='orange', marker='s', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

model.save('simple_rnn_imdb_model.h5')
from tensorflow.keras.models import load_model
model = load_model('simple_rnn_imdb_model.h5')
model.summary()

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

sample_review = X_test[1]
sample_review
len(sample_review)

len(sample_review.reshape(1, -1))
prediction = model.predict(sample_review.reshape(1, -1))
prediction

sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
print(f"Predicted sentiment:", sentiment)
print(f"Prediction confidence:", prediction[0][0])

print(f"Actual label:", "Positive" if y_test[1] == 1 else "Negative")
