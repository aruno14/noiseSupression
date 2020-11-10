
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

clear_files = glob.glob("data/clean/*.wav")
noise_files = glob.glob("data/noise/*.wav")

for clear_filepath in clear_files:
    audio_binary = tf.io.read_file(clear_filepath)
    audio_clear, audioSR_clear = tf.audio.decode_wav(audio_binary)
    audio_clear = tf.squeeze(audio_clear, axis=-1)
    clear_file_code = clear_filepath.replace("data/clean/", "").replace(".wav", "")
    print("Load:", clear_filepath, clear_file_code, audioSR_clear)
    for noise_filepath in noise_files:
        audio_binary = tf.io.read_file(noise_filepath)
        audio_noise, audioSR_noise = tf.audio.decode_wav(audio_binary)
        noise_file_code = noise_filepath.replace("data/noise/", "").replace(".wav", "")
        print("Load:", noise_filepath, noise_file_code, audioSR_noise)
        audio_noise = tf.squeeze(audio_noise, axis=-1)
        #plt.figure("Oscillo clear")
        #plt.plot(audio_clear.numpy())
        #plt.show()
        #plt.figure("Oscillo noise")
        #plt.plot(audio_noise.numpy())
        #plt.show()
        while len(audio_noise) < len(audio_clear):
            audio_noise = tf.concat([audio_noise, audio_noise], -1)
        audio_noise = audio_noise[0:len(audio_clear)]
        audio_noise = audio_noise * 0.05
        audio_with_noise = tf.math.add(audio_clear, audio_noise)
        #print(audio_clear.shape)
        #print(audio_noise.shape)
        #print(audio_with_noise.shape)
        #plt.figure("Oscillo fusion")
        #plt.plot(audio_with_noise.numpy())
        #plt.show()

        audio_with_noise = tf.expand_dims(audio_with_noise, axis=-1)

        audio_string = tf.audio.encode_wav(audio_with_noise, sample_rate=audioSR_clear)
        noisy_filepath = "data/noisy/"+clear_file_code+"_"+noise_file_code+".wav"
        tf.io.write_file(noisy_filepath, contents=audio_string)
        print("File saved:", noisy_filepath)

        audio_string = tf.audio.encode_wav(tf.expand_dims(audio_clear, axis=-1), sample_rate=audioSR_clear)
        clear_filepath = "data/clear/"+clear_file_code+"_"+noise_file_code+".wav"
        tf.io.write_file(clear_filepath, contents=audio_string)
