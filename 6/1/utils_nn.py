# coding=utf-8

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

# 第一种残差模型
def resNetBlockV1(inputs, size=64):
    layer = slim.conv2d(inputs, size, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer,  size, [3,3], normalizer_fn=slim.batch_norm, activation_fn=None)
    return tf.nn.relu(inputs + layer) 

# 第二种残差模型
def resNetBlockV2(inputs, size=64):
    layer = slim.conv2d(inputs, size,   [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer,  size,   [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer,  size*4, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    return tf.nn.relu(inputs + layer)   

def resNet18(layer, isPoolSize=True):
    if isPoolSize:
        stride = 2
        padding = "VALID"
    else:
        stride = 1
        padding = "SAME"
    with slim.arg_scope([slim.max_pool2d, slim.avg_pool2d], stride=stride, padding=padding):
        layer = slim.conv2d(layer, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        
        for i in range(2):
            layer = resNetBlockV1(layer, 64)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        for i in range(2):
            layer = resNetBlockV1(layer, 128)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)        
        for i in range(2):
            layer = resNetBlockV1(layer, 256)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        for i in range(2):
            layer = resNetBlockV1(layer, 512)
        return layer

def resNet34(layer, isPoolSize=True):
    if isPoolSize:
        stride = 2
        padding = "VALID"
    else:
        stride = 1
        padding = "SAME"
    with slim.arg_scope([slim.max_pool2d, slim.avg_pool2d], stride=stride, padding=padding):
        layer = slim.conv2d(layer, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 

        for i in range(3):
            layer = resNetBlockV1(layer, 64)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        for i in range(4):
            layer = resNetBlockV1(layer, 128)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)        
        for i in range(6):
            layer = resNetBlockV1(layer, 256)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        for i in range(3):
            layer = resNetBlockV1(layer, 512)
        return layer

def resNet50(layer, isPoolSize=True, stride=2):
    if isPoolSize:
        stride = stride
        padding = "VALID"
    else:
        stride = 1
        padding = "SAME"
    with slim.arg_scope([slim.max_pool2d, slim.avg_pool2d], stride=stride, padding=padding):
        layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        for i in range(3):
            layer = resNetBlockV2(layer, 64)
        layer = slim.max_pool2d(layer, [2, 2])

        layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        for i in range(4):
            layer = resNetBlockV2(layer, 128)
        layer = slim.max_pool2d(layer, [2, 2])
        half_layer = layer

        layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)        
        for i in range(6):
            layer = resNetBlockV2(layer, 256)
        layer = slim.max_pool2d(layer, [2, 2])

        layer = slim.conv2d(layer, 2048, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        for i in range(3):
            layer = resNetBlockV2(layer, 512)
        return layer, half_layer    

def resNet101(layer, isPoolSize=True):
    if isPoolSize:
        stride = 2
        padding = "VALID"
    else:
        stride = 1
        padding = "SAME"
    with slim.arg_scope([slim.max_pool2d, slim.avg_pool2d], stride=stride, padding=padding):
        for i in range(3):
            layer = resNetBlockV2(layer, 64)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        for i in range(4):
            layer = resNetBlockV2(layer, 128)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)        
        for i in range(23):
            layer = resNetBlockV2(layer, 256)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 2048, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        for i in range(3):
            layer = resNetBlockV2(layer, 512)
        return layer   

def resNet152(layer, isPoolSize=True):
    if isPoolSize:
        stride = 2
        padding = "VALID"
    else:
        stride = 1
        padding = "SAME"
    with slim.arg_scope([slim.max_pool2d, slim.avg_pool2d], stride=stride, padding=padding):
        for i in range(3):
            layer = resNetBlockV2(layer, 64)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        for i in range(8):
            layer = resNetBlockV2(layer, 128)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)        
        for i in range(36):
            layer = resNetBlockV2(layer, 256)
        layer = slim.max_pool2d(layer, [3, 3])

        layer = slim.conv2d(layer, 2048, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        for i in range(3):
            layer = resNetBlockV2(layer, 512)
        return layer 

# 参考 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py
def INCEPTIONV3(inputs):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        # mixed_5b
        net    = inputs
        layer0 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,    48, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 64, [5,5], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net, [3,3])
        layer3 = slim.conv2d(layer3, 32, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_5c => 288
        layer0 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,    48, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 64, [5,5], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)        
        # mixed_5d => 288
        layer0 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,    48, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 64, [5,5], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net = tf.concat([layer0, layer1, layer2, layer3], 3)   
        # mixed_6a => 768
        layer0 = slim.conv2d(net,   384, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.max_pool2d(net,  [3,3])
        net    = tf.concat([layer0, layer1, layer2], 3)
        # mixed_6b => 768
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,128, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,   128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,128, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,128, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,128, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_6c => 768
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,160, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,   160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_6d => 768
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,160, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,   160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_6e => 768
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_7a => 1280
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer0 = slim.conv2d(layer0,320, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
            
        net    = tf.concat([layer0, layer1, layer2], 3)
        # mixed_7b => 2048
        layer0 = slim.conv2d(net,   320, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   384, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1_1 = slim.conv2d(layer1,384, [1,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1_2 = slim.conv2d(layer1,384, [3,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1    = tf.concat([layer1_1, layer1_2], 3)
        layer2 = slim.conv2d(net,   448, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,   384, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2_1 = slim.conv2d(layer2,384, [1,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2_2 = slim.conv2d(layer2,384, [3,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2    = tf.concat([layer2_1, layer2_2], 3)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_7c => 2048
        layer0 = slim.conv2d(net,   320, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   384, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1_1 = slim.conv2d(layer1,384, [1,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1_2 = slim.conv2d(layer1,384, [3,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1    = tf.concat([layer1_1, layer1_2], 3)
        layer2 = slim.conv2d(net,   448, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,   384, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2_1 = slim.conv2d(layer2,384, [1,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2_2 = slim.conv2d(layer2,384, [3,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2    = tf.concat([layer2_1, layer2_2], 3)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
    return net

def pix2pix_g(inputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],  kernel_size=[4, 4],  stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm):
        # Encoder 
        encoder_activations=[]
        layer = slim.conv2d(inputs, 64, normalizer_fn=None)
        encoder_activations.append(layer)
        for cnn in (128,256,512,512,512):
            layer = slim.conv2d(layer, cnn)
            encoder_activations.append(layer)
        layer = slim.conv2d(layer, 512, normalizer_fn=None, activation_fn=None,)
        encoder_activations.append(layer)
        # Decoder 
        layer = tf.nn.relu(layer)
        layer = slim.conv2d_transpose(layer, 512, normalizer_fn=None, activation_fn=None,)
        layer = tf.concat([layer, encoder_activations[-1]], 3)
        for i, cnn in enumerate((512,512,512,256,128)):
            layer = slim.conv2d_transpose(layer, cnn)
            layer = tf.concat([layer, encoder_activations[-i-2]], 3)
        layer = slim.conv2d_transpose(layer, 64, normalizer_fn=None)
        layer = tf.concat([layer, encoder_activations[0]], 3)
        layer = slim.conv2d(layer, 1, normalizer_fn=None, activation_fn=None)
        layer = tf.tanh(layer)
        return layer

def pix2pix_d(inputs):
    with slim.arg_scope([slim.conv2d],  kernel_size=[4, 4],  stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm):
        layer = slim.conv2d(inputs, 64, normalizer_fn=None)
        for cnn in (64,64,64,0,128,128,128,128,0,256,256,256,256,256,256,0,512,512,512):
            if cnn == 0:
                layer = slim.max_pool2d(layer,  [2,2])
            else:
                layer = slim.conv2d(layer, cnn)
        layer = slim.conv2d(layer, 256, stride=1)
        layer = slim.conv2d(layer, 1, stride=1, normalizer_fn=None, activation_fn=None)
        # layer = tf.sigmoid(layer)
        return layer

# inputs 512 * 512
def pix2pix_g2(layer, dropout=False): 
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], kernel_size=[4, 4], stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm):
        # Encoder 
        encoder_activations=[]
        for cnn in (64,128,256,512,512,512,512,512):
            layer = slim.conv2d(layer, cnn)
            encoder_activations.append(layer)

        half_layer = layer

        # # 加了随机噪声也许会更好一些？
        # batch_size = tf.shape(inputs)[0]
        # embeddings = tf.get_variable("E", [batch_size, 1, 1, 512], tf.float32, tf.random_normal_initializer())
        # layer = tf.concat([layer, embeddings], 3)

        # Decoder 
        for i, cnn in enumerate((512,512,512,512,256,128,64)):
            layer = slim.conv2d_transpose(layer, cnn)
            if dropout and i in [0,1,2]:
                layer = tf.nn.dropout(layer, 0.5)
            # print(layer.shape)               
            layer = tf.concat([layer, encoder_activations[-i-2]], 3)
        layer = slim.conv2d_transpose(layer, 1, normalizer_fn=None, activation_fn=None)
        layer = tf.tanh(layer)
        return layer, half_layer

def pix2pix_d2(layer):
    with slim.arg_scope([slim.conv2d], kernel_size=[4, 4], stride=[2,1], activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm):
        for i, cnn in enumerate((64,64,64,128,128,128,128,256,256,256,256,256,256,512,512,512)):
            if i % 2 ==0:
                layer = slim.conv2d(layer, cnn, kernel_size=[3, 3], stride=1) 
            else:
                layer = slim.conv2d(layer, cnn)
    layer = slim.conv2d(layer, 1000, kernel_size=[1, 1], stride=1, normalizer_fn=None, activation_fn=None)
    layer = tf.reduce_mean(layer, [1, 2], keep_dims=True)
    layer = slim.flatten(layer)
    layer = slim.fully_connected(layer, 1000)
    layer = slim.fully_connected(layer, 1)
    # layer = tf.sigmoid(layer)
    return layer