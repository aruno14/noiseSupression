# NoiseSupression
A simple Noise Suppression Model based on RNN and implemented with TensorFlow.

More details: https://aruno14.medium.com/when-your-boss-say-make-noise-suppression-system-8e82ee84afd7

## Dataset
* https://github.com/breizhn/DNS-Challenge (~300Go)

Noisy and Clear data are created using `create_noisy.py`.

## Model comparison
Models have been test on a very small subdataset (about 20 files). The MSE values are calculated on the training data.

| Model  | MSE | Size |
| --------  | ------------------- | --------------------- |
| No model | 1.34     | - | 
| Dense      | 1.22 | 854Ko | 
| Dense_gain      | 0.75 | 862Ko | 
| Dense_image      | 0.68 | 367Mo | 
| Lstm_simple      | 1.31 | 8Mo | 
| Lstm_sequence      | 0.56 | 8Mo | 
| Lstm_image      | 0.59 | 5Go | 
