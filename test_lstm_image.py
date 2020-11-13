import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Multiply, TimeDistributed, Reshape, MaxPooling2D, Conv2D, Dropout, Flatten
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import backend as K
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

block_length = 0.050#->500ms
voice_max_length = int(0.5/block_length)#->10s
frame_length = 512
image_width = 128
model_name = "noise_model_lstm_image"
batch_size = 32
epochs = 5

print("voice_max_length:", voice_max_length)
def audioToTensor(filepath:str):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    frame_step = int(audioSR * 0.008)#16000*0.008=128
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_image = tf.math.imag(spectrogram)
    spect_real = tf.math.real(spectrogram)
    spect_sign = tf.sign(spect_real)
    spect_real = tf.abs(spect_real)
    partsCount = len(range(0, len(spectrogram)-image_width, image_width))
    parts = np.zeros((partsCount, image_width, int(frame_length/2+1)))
    for i, p in enumerate(range(0, len(spectrogram)-image_width, image_width)):
        part = spect_real[p:p+image_width]
        parts[i] = part
    return spect_real, spect_image, spect_sign, audioSR, parts

def spectToOscillo(spect_real, spect_sign, spect_image, audioSR):
    frame_step = int(audioSR * 0.008)
    spect_real*=spect_sign
    spect_all = tf.complex(spect_real, spect_image)
    return tf.signal.inverse_stft(spect_all, frame_length=frame_length, frame_step=frame_step, window_fn=tf.signal.inverse_stft_window_fn(frame_step))

clear_files = glob.glob("data/clear/*.wav")

x_train = []
x_train_count = 0
for i, path_clear in enumerate(clear_files):
    spectNoisy, _, _, audioNoisySR, partsNoise = audioToTensor(path_clear)
    x_train.append((path_clear, len(partsNoise)-voice_max_length))
    x_train_count+=len(partsNoise)//voice_max_length
print("x_train_count:", x_train_count)

class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_train, x_train_count, batch_size):
        self.x_train= x_train
        self.x_train_count = x_train_count
        self.batch_size = batch_size
    def __len__(self):
        return self.x_train_count//self.batch_size
    def __getitem__(self, idx):
        batch_x_train = np.zeros((batch_size, voice_max_length, image_width, int(frame_length/2+1)))
        batch_y_train = np.zeros((batch_size, voice_max_length, image_width, int(frame_length/2+1)))
        current_size = 0
        while current_size < batch_size:
            path_clear, _ = self.x_train[(idx * self.batch_size + current_size)%len(clear_files)]
            
            path_noisy = path_clear.replace("clear", "noisy")
            spectNoisy, _, _, audioNoisySR, partsNoise = audioToTensor(path_noisy)
            spectClear, _, _, audioClearSR, partsClear = audioToTensor(path_clear)
            for k in range(0, min(len(partsNoise), len(partsClear))//voice_max_length):
                batch_x_train[current_size] = partsNoise[k*voice_max_length:(k+1)*voice_max_length]
                batch_y_train[current_size] = partsClear[k*voice_max_length:(k+1)*voice_max_length]
                current_size+=1
                if current_size>=batch_size:
                    break
        return batch_x_train, batch_y_train

print('Build model...')

if os.path.exists(model_name):
    print("Load: " + model_name)
    model = load_model(model_name)
else:
    main_input = Input(shape=(voice_max_length, image_width, int(frame_length/2+1)), name='main_input')
    x = main_input
    x = TimeDistributed(Reshape((image_width, int(frame_length/2+1), 1)))(x)
    x = TimeDistributed(preprocessing.Resizing(image_width//2, int(frame_length/2+1)//2))(x)
    x = TimeDistributed(Conv2D(34, 3, activation='relu'))(x)
    x = TimeDistributed(Conv2D(64, 3, activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D())(x)
    x = TimeDistributed(Dropout(0.1))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)(x)
    x = Dense(int(frame_length/2+1), activation='sigmoid')(x)
    x = Reshape((voice_max_length, 1, int(frame_length/2+1)))(x)
    x = Multiply()([x, main_input])
    model = Model(inputs=main_input, outputs=x)
    tf.keras.utils.plot_model(model, to_file='model_lstm_image.png', show_shapes=True)
model.compile(loss='mse', metrics='mse', optimizer='adam')#Adam, SGD, Adagrad

print('Train...')
history = model.fit(MySequence(x_train, x_train_count, batch_size), epochs=epochs, steps_per_epoch=x_train_count//batch_size)
model.save(model_name)

metrics = history.history
plt.plot(history.epoch, metrics['mse'])
plt.legend(['mse'])
plt.savefig("learning-lstm_image.png")
plt.show()
plt.close()

print('Evaluate no prediction...')
total_loss = 0
for i, path_clear in enumerate(clear_files):
    path_noisy = path_clear.replace("clear", "noisy")
    spectNoisy, _, _, audioNoisySR, partsNoise = audioToTensor(path_noisy)
    spectClear, _, _, audioClearSR, partsClear = audioToTensor(path_clear)
    loss = np.mean(tf.keras.losses.mean_squared_error(spectClear, spectNoisy).numpy())
    total_loss+=loss
    print(path_noisy, "->", loss)
print("total_loss:", total_loss/len(clear_files))

print('Evaluate...')
total_loss = 0
for i, path_clear in enumerate(clear_files):
    path_noisy = path_clear.replace("clear", "noisy")
    spectNoisy, _, _, audioNoisySR, partsNoise = audioToTensor(path_noisy)
    spectClear, _, _, audioClearSR, partsClear = audioToTensor(path_clear)
    
    input = np.zeros((len(partsNoise)//voice_max_length, voice_max_length, image_width, int(frame_length/2+1)))
    for i in range(0, len(partsNoise)//voice_max_length):
        input[i] = partsNoise[i*voice_max_length:(i+1)*voice_max_length]
    result = model.predict(input)
    result = np.reshape(result, (result.shape[0]*result.shape[1]*result.shape[2], result.shape[3]))
    loss = np.mean(tf.keras.losses.mean_squared_error(spectClear[0:len(result)], result).numpy())
    total_loss+=loss
    print(path_noisy, "->", loss)
print("total_loss:", total_loss/len(clear_files))

for test_path in [('data/noisy/book_00000_chp_0009_reader_06709_0_---1_cCGK4M.wav'), ('data/noisy/book_00000_chp_0009_reader_06709_1_---1_cCGK4M.wav'), ('data/noisy/book_00000_chp_0009_reader_06709_16_---1_cCGK4M.wav')]:
    print("test_string: ", test_path)
    spect_real, spect_image, spect_sign, audioSR, parts = audioToTensor(test_path)
    plt.figure("Spect: " + test_path)
    plt.imshow(tf.math.log(spect_real).numpy())
    plt.show()
    print("spect_real:", spect_real)
    input = np.zeros((len(parts)//voice_max_length, voice_max_length, image_width, int(frame_length/2+1)))
    print("len(spect_real):", len(spect_real))
    print("test_audio_abs.shape:", input.shape)
    for i in range(0, len(parts)//voice_max_length):
        input[i] = parts[i*voice_max_length:(i+1)*voice_max_length]
    print("input.shape:", input.shape)
    result = model.predict(input)
    result = np.reshape(result, (result.shape[0]*result.shape[1]*result.shape[2], result.shape[3]))
    print("result_gain.shape:", result.shape)
    print("spect_real:", spect_real)
    plt.figure("Spect: " + test_path)
    plt.imshow(tf.math.log(result).numpy())
    plt.show()
    print("diff:", K.sum(K.abs(spect_real[0:len(result)] - result)))
    oscillo = spectToOscillo(spect_real=result, spect_sign=spect_sign[0:len(result)], spect_image=spect_image[0:len(result)], audioSR=audioSR)
    oscillo=tf.expand_dims(oscillo, axis=-1)
    audio_string = tf.audio.encode_wav(oscillo, sample_rate=audioSR)
    clear_filepath = test_path.replace("data/noisy/", "test_")
    tf.io.write_file(clear_filepath, contents=audio_string)
