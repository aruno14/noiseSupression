import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Multiply
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

block_length = 0.050#->500ms
voice_max_length = int(0.5/block_length)#->3s
frame_length = 512
model_name = "noise_model_dense_gain"
batch_size = 32
epochs=1

print("voice_max_length:", voice_max_length)
def audioToTensor(filepath:str):
    #print("audioToTensor:filepath:", filepath)
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    frame_step = int(audioSR * 0.008)#16000*0.008=128
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)#->31hz, si 512 -> 64hz
    spect_image = tf.math.imag(spectrogram)
    spect_real = tf.math.real(spectrogram)
    spect_sign = tf.sign(spect_real)
    spect_real = tf.abs(spect_real)
    spect_real = tf.math.log(spect_real)/tf.math.log(tf.constant(10, dtype=tf.float32))*20#decibels
    spect_real = tf.where(tf.math.is_nan(spect_real), tf.zeros_like(spect_real), spect_real)
    spect_real = tf.where(tf.math.is_inf(spect_real), tf.zeros_like(spect_real), spect_real)
    return spect_real, spect_image, spect_sign, audioSR

def spectToOscillo(spect_real, spect_sign, spect_image, audioSR):
    frame_step = int(audioSR * 0.008)
    spect_real = pow(10, spect_real/20)#power value
    spect_real*=spect_sign
    spect_all = tf.complex(spect_real, spect_image)
    inverse_stft = tf.signal.inverse_stft(spect_all, frame_length=frame_length, frame_step=frame_step, window_fn=tf.signal.inverse_stft_window_fn(frame_step))
    return inverse_stft

clear_files = glob.glob("data/clear/*.wav")

x_train = []
x_train_count = 0
for i, path_clear in enumerate(clear_files):
    spectNoisy, _, _, audioNoisySR  = audioToTensor(path_clear)
    x_train.append((path_clear, len(spectNoisy)))
    x_train_count+=len(spectNoisy)
print("x_train_count:", x_train_count)

class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_train, x_train_count, batch_size):
        self.x_train= x_train
        self.x_train_count = x_train_count
        self.batch_size = batch_size
    def __len__(self):
        return self.x_train_count//self.batch_size
    def __getitem__(self, idx):
        batch_x_train = np.zeros((batch_size, int(frame_length/2+1)))
        batch_y_train = np.zeros((batch_size, int(frame_length/2+1)))
        current_size = 0
        while current_size < batch_size:
            path_clear, _ = self.x_train[(idx * self.batch_size + current_size)%len(clear_files)]
            path_noisy = path_clear.replace("clear", "noisy")
            spectNoisy, _, _, _ = audioToTensor(path_noisy)
            spectClear, _, _, _ = audioToTensor(path_clear)
            for k in range(0, len(spectNoisy)):
                batch_x_train[current_size] = spectNoisy[k]
                batch_y_train[current_size] = spectClear[k]
                current_size+=1
                if current_size>=batch_size:
                    break
        return batch_x_train, batch_y_train

print('Build model...')
if os.path.exists(model_name):
    print("Load: " + model_name)
    model = load_model(model_name)
else:
    main_input = Input(shape=(int(frame_length/2+1)), name='main_input')
    x = main_input
    x = Dense(int(frame_length/2+1), activation='sigmoid')(x)
    x = Multiply()([x, main_input])
    model = Model(inputs=main_input, outputs=x)
    tf.keras.utils.plot_model(model, to_file='model_dense_gain.png', show_shapes=True)
model.compile(loss='mse', metrics='mse', optimizer='adam')

print('Train...')
history = model.fit(MySequence(x_train, x_train_count, batch_size), epochs=epochs, steps_per_epoch=x_train_count//batch_size)
model.save(model_name)

metrics = history.history
plt.plot(history.epoch, metrics['mse'])
plt.legend(['mse'])
plt.savefig("learning-dense_gain.png")
plt.show()
plt.close()

print('Evaluate...')
total_loss = 0
for i, path_clear in enumerate(clear_files):
    path_noisy = path_clear.replace("clear", "noisy")
    spectNoisy, _, _, _ = audioToTensor(path_noisy)
    spectClear, _, _, _ = audioToTensor(path_clear)
    result = model.predict(spectNoisy)
    loss = np.mean(tf.keras.losses.mean_squared_error(spectClear, result).numpy())
    total_loss+=loss
    print(path_noisy, "->", loss)
print("total_loss:", total_loss/len(clear_files))

for test_path in [('data/noisy/book_00000_chp_0009_reader_06709_0_---1_cCGK4M.wav'), ('data/noisy/book_00000_chp_0009_reader_06709_1_---1_cCGK4M.wav')]:
    print("test_string: ", test_path)
    spect_real, spect_image, spect_sign, audioSR = audioToTensor(test_path)
    print("spect_real:", spect_real)
    result = model.predict(spect_real)
    print("result_gain.shape:", result.shape)
    oscillo = spectToOscillo(spect_real=result, spect_sign=spect_sign, spect_image=spect_image, audioSR=16000)
    oscillo=tf.expand_dims(oscillo, axis=-1)
    audio_string = tf.audio.encode_wav(oscillo, sample_rate=audioSR)
    clear_filepath = test_path.replace("data/noisy/", "test_")
    tf.io.write_file(clear_filepath, contents=audio_string)
