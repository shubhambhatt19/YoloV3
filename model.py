import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Concatenate
from IPython import embed
import config as cfg
from tensorflow.keras import layers, models

from config import config


def CNNBlock(x, filters, kernel_size, stride, use_bn_act=True):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same' if kernel_size == 3 else 'valid', use_bias=not use_bn_act)(x)
    if use_bn_act:
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
    return x

def ResidualBlock(x, channels,use_residual=False, num_repeats=None):
    for _ in range(num_repeats):
        shortcut = x
        x = CNNBlock(x, channels // 2, 1, 1)
        x = CNNBlock(x, channels, 3, 1)
        if use_residual:
            x = layers.Add()([shortcut, x])
    return x

def ScalePrediction(x, in_channels, num_classes):
    x = CNNBlock(x, 2 * in_channels, 3, 1)
    x = CNNBlock(x, (num_classes + 5) * 3, 1, 1, use_bn_act=False)
    x = layers.Reshape((3,num_classes + 5, x.shape[1], x.shape[2]))(x)
    x = layers.Permute((1,3,4,2))(x)
    return x

def YOLOv3(input_shape=(416, 416, 3), num_classes=20):
    inputs = layers.Input(shape=input_shape)
    print("#######################")
    print("inputs", inputs)
    print("#######################")

    x = inputs
    out = []

    ##################### darknet starts ####################
    filters, kernel_size, stride = config[0]
    x = CNNBlock(x, filters, kernel_size, stride)

    filters, kernel_size, stride = config[1]
    x = CNNBlock(x, filters, kernel_size, stride)

    _, num_repeats = config[2]    
    x = ResidualBlock(x, channels=x.shape[-1], num_repeats=num_repeats)

    filters, kernel_size, stride = config[3]
    x = CNNBlock(x, filters, kernel_size, stride)

    _, num_repeats = config[4]    
    x = ResidualBlock(x, channels=x.shape[-1], num_repeats=num_repeats)

    filters, kernel_size, stride = config[5]
    x = CNNBlock(x, filters, kernel_size, stride)

    _, num_repeats = config[6]    
    x = ResidualBlock(x, channels=x.shape[-1], num_repeats=num_repeats)
    res1 = x

    filters, kernel_size, stride = config[7]
    x = CNNBlock(x, filters, kernel_size, stride)

    _, num_repeats = config[8]    
    x = ResidualBlock(x, channels=x.shape[-1], num_repeats=num_repeats)
    res2 = x

    filters, kernel_size, stride = config[9]
    x = CNNBlock(x, filters, kernel_size, stride)

    _, num_repeats = config[10]    
    x = ResidualBlock(x, channels=x.shape[-1], num_repeats=num_repeats)
    ##################### darknet end ####################

    ########### (512, 1, 1) ###########
    filters, kernel_size, stride = config[11]
    x = CNNBlock(x, filters, kernel_size, stride)

    #######(1024, 3, 1),###################
    filters, kernel_size, stride = config[12]
    x = CNNBlock(x, filters, kernel_size, stride)

    ####### "S",########
    x = ResidualBlock(x, channels=x.shape[-1], num_repeats=1)
    
    x = CNNBlock(x, x.shape[-1]//2, kernel_size=1, stride = 1)
    out_ = ScalePrediction(x, x.shape[-1] // 2, num_classes)

    out.append(out_)

    ########## (256, 1, 1), #########
    filters, kernel_size, stride = config[14]
    x = CNNBlock(x, filters, kernel_size, stride)

    ############ "U", ##############
    print("before upsampling", x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    print("after upsampling", x)

    x = tf.keras.layers.Concatenate()([x, res2])

    ############# (256, 1, 1), ##############
    filters, kernel_size, stride = config[16]
    x = CNNBlock(x, filters, kernel_size, stride)

    ############### (512, 3, 1), ############
    filters, kernel_size, stride = config[17]
    x = CNNBlock(x, filters, kernel_size, stride)


    ################### "S", ###################
    x = ResidualBlock(x, channels=x.shape[-1], num_repeats=1)
    
    x = CNNBlock(x, x.shape[-1]//2, kernel_size=1, stride = 1)
    out_ = ScalePrediction(x, x.shape[-1] // 2, num_classes)

    out.append(out_)

    ############# (128, 1, 1)##########
    filters, kernel_size, stride = config[19]
    x = CNNBlock(x, filters, kernel_size, stride)

    ############## "U", ##############
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, res1])

    ############# (128, 1, 1), ##############
    filters, kernel_size, stride = config[21]
    x = CNNBlock(x, filters, kernel_size, stride)

    ############### (256, 3, 1), ############
    filters, kernel_size, stride = config[22]
    x = CNNBlock(x, filters, kernel_size, stride)

    ############### "S" ###################
    x = ResidualBlock(x, channels=x.shape[-1], num_repeats=1)
    x = CNNBlock(x, x.shape[-1]//2, kernel_size=1, stride = 1)
    out_ = ScalePrediction(x, x.shape[-1] // 2, num_classes)
    out.append(out_)
    
    model = models.Model(inputs, out)
    return model

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=num_classes)
    x = tf.random.normal((2, IMAGE_SIZE, IMAGE_SIZE, 3))
    out = model(x)
    print(model.summary())
    assert model(x)[0].shape == (2, 13, 13, 3, num_classes + 5)
    assert model(x)[1].shape == (2, 26, 26, 3, num_classes + 5)
    assert model(x)[2].shape == (2, 52, 52, 3, num_classes + 5)
    print("Success!")