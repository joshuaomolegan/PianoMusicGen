from preprocessing import inputs, targets
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM

model = Sequential()

model.add(LSTM(512, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(512))
model.add(Dropout(0.6))
model.add(Dense(256))
model.add(Dropout(0.6))
model.add(Dense(256))
model.add(Dropout(0.6))
model.add(Dense(256))
model.add(Dropout(0.6))
model.add(Dense(88, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

model.fit(inputs, targets, epochs=5, batch_size=512, verbose=2)

model.save("model.hdf5")