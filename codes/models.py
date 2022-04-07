from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Input, Concatenate, Average,Conv1DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf


#_______________________________________________
# Encoder (Downsampler) 
#_______________________________________________
def downsample(filters, size, apply_batchnorm=True):

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv1D(filters, size, strides=2, padding='same'))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU(0.25))

    return result


#_______________________________________________
# Decoder (Upsampler) 
#_______________________________________________
def upsample(filters, size, apply_dropout=False):

    result = tf.keras.Sequential()
    result.add(Conv1DTranspose(filters, size, strides=2 , padding='same'))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.LeakyReLU(0.25))

    return result


################################################
#_______________________________________________
# Signal to Signal Model UNET APPROACH (Deep CNN - 12 layers)
# Trainable params: 79,121
#_______________________________________________
################################################
def sig2sig_unet(win_size, channel = 1):
    inputs = tf.keras.layers.Input(shape=[win_size,channel])

    down_stack = [
    downsample(16, 9, apply_batchnorm=False), # (bs, 8000, 1)
    downsample(16, 9), # (bs, 4000, 1)
    downsample(32, 6), # (bs, 2000, 1)
    downsample(32, 6), # (bs, 1000, 1)
    downsample(64, 3), # (bs, 500, 1)
    downsample(64, 3), # (bs, 250, 1)
    ]

    up_stack = [
    upsample(64, 3, apply_dropout=True), # (bs, 250, 1)
    upsample(32, 3), # (bs, 500, 1)
    upsample(32, 6), # (bs, 1000, 1)
    upsample(16, 6), # (bs, 2000, 1)
    upsample(16, 9), # (bs, 4000, 1)
    ]

    OUTPUT_CHANNELS = 1
    last = Conv1DTranspose(OUTPUT_CHANNELS, 9,
                                            strides=2,
                                            padding='same',
        
                                            activation='sigmoid') 

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


################################################
#_______________________________________________
# Signal to Signal Model Encoder-Decoder (12 layers - Without skip connections)   
# Trainable params: 61,313
#_______________________________________________
################################################
def sig2sig_cnn(win_size, channel = 1):

    inputs = tf.keras.layers.Input(shape=[win_size,channel])

    down_stack = [
    downsample(16, 9, apply_batchnorm=False), # (bs, 8000, 1)
    downsample(16, 9), # (bs, 4000, 1)
    downsample(32, 6), # (bs, 2000, 1)
    downsample(32, 6), # (bs, 1000, 1)
    downsample(64, 3), # (bs, 500, 1)
    downsample(64, 3), # (bs, 250, 1)
    ]

    up_stack = [
    upsample(64, 3, apply_dropout=True), # (bs, 250, 1)
    upsample(32, 3), # (bs, 500, 1)
    upsample(32, 6), # (bs, 1000, 1)
    upsample(16, 6), # (bs, 2000, 1)
    upsample(16, 9), # (bs, 4000, 1)
    ]

    OUTPUT_CHANNELS = 1
    last = Conv1DTranspose(OUTPUT_CHANNELS, 9,
                                            strides=2,
                                            padding='same',
        
                                            activation='sigmoid') # (bs, 8000, 1)

    x = inputs

    # Downsampling through the model
    #skips = []
    for down in down_stack:
        x = down(x)

    # Upsampling 
    for up in up_stack:
        x = up(x)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



if __name__ == "__main__":
    pass









    