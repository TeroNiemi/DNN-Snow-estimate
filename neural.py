import tensorflow as tf
import pandas as pd
import numpy as np
import os


# Luetaan treenidata
train_data = pd.read_csv('train.csv', delimiter=';')

# Siistitään data pilkut pisteiksi ja floatiksi
train_data['LowTemp'] = train_data['LowTemp'].str.replace(',', '.').astype(float)
train_data['HighTemp'] = train_data['HighTemp'].str.replace(',', '.').astype(float)
train_data['SnowThickness'] = train_data['SnowThickness'].astype(float)

# Irrotetaan input ja output
train_x = np.array(train_data[['LowTemp', 'HighTemp']], dtype=np.float32)
train_y = np.array(train_data['SnowThickness'], dtype=np.float32)

# Normalisoidaan
train_x = (train_x - np.mean(train_x)) / np.std(train_x)

# % opetusdataksi
train_size = int(len(train_x) * 0.7)
train_x, test_x = train_x[:train_size], train_x[train_size:]
train_y, test_y = train_y[:train_size], train_y[train_size:]

# Määritellään neuroverkon malli, piilokerrokset ja neuronit

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    
    tf.keras.layers.Dense(1, activation='linear')
])



# Treeni vai testaus?
if os.path.exists('snow_model.h5'):
    while True:
        choice = input('Do you want to train the model again? (y/n): ').lower()
        if choice == 'n':
            
            model = tf.keras.models.load_model('\snow_model.h5')
            break
        elif choice == 'y':

            # Treenaa ja saveta
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=2)
          
            model.save('snow_model.h5')
            break
        else:
            print('Invalid choice, please enter y or n.')
else:
    # Treenaa ja saveta
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=2)
    
    model.save('snow_model.h5')

# Testataan
test_loss = model.evaluate(test_x, test_y, verbose=0)
print(f'Test loss: {test_loss}')

# Ennuta uuden datan avulla
input_data = pd.read_csv('input.csv', delimiter=';')
input_data['LowTemp'] = input_data['LowTemp'].str.replace(',', '.').astype(float)
input_data['HighTemp'] = input_data['HighTemp'].str.replace(',', '.').astype(float)
new_data = np.array(input_data[['LowTemp', 'HighTemp']], dtype=np.float32)
new_data = (new_data - np.mean(train_x)) / np.std(train_x)
predictions = model.predict(new_data)
predictions = np.maximum(predictions, 0)
print(f'Predictions: {predictions.flatten()}')
