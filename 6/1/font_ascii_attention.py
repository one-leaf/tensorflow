# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import utils
import time
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception
import math
import urllib,json,io
import utils_pil, utils_font, utils_nn
import operator
from collections import deque
import font_dataset

curr_dir = os.path.dirname(__file__)

CLASSES_NUMBER = font_dataset.CLASSES_NUMBER
NULL_CODE = font_dataset.CLASSES_NUMBER - 1 
# 采用注意力模型，需要确定下图片的大小和最大字数
# 不足高宽的需要补齐
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 4096
SEQ_LENGTH  = 256

#初始化学习速率
LEARNING_RATE_INITIAL = 1e-4
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
LSTM_UNITS_NUMBER = 256

REPORT_STEPS = 500

BATCHES = 50
BATCH_SIZE = 1
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 4
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_ascii_attention"

# CNN特征采集
# 输入[B 32 4096 1] ==> [B 1 1024 256]
def CNN(inputs):
    with tf.variable_scope("CNN"):
        # layer = slim.conv2d(inputs, 64, [8,8], [2,4], normalizer_fn=slim.batch_norm, activation_fn=None) 
        # layer [B H//2 W//4 64]
        # tf.summary.image('zoom', tf.transpose (layer, [3, 1, 2, 0]), max_outputs=6)
        # layer = utils_nn.resNet50(layer, True, [2,1]) 
        # [N H//32 W 2048]
        # tf.summary.image('2_res50', tf.transpose (layer, [3, 1, 2, 0]), max_outputs=6)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=True):
                layer, _ = inception.inception_v3_base(inputs, final_endpoint="Mixed_5d")

        # 直接将网络拉到256 [N 1 W 256]
        with tf.variable_scope("Normalize"):
            layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
            return layer

# 增加坐标信息，增加的个数为 embedd_size
# max_width_height, embedd_size
# max_width_height 为缩放后的 w 的最大宽度，实际上的最大图片宽度为 max_width_height * 4
# 输入[B 1 W 256] ==> [B 1 W 288]
def Coordinates(inputs):
    with tf.variable_scope("Coordinates"):
        # 这个是官方办法，但这个矩阵太大了，直接由 256 => 1281 所以改了
        # _, h, w, _ = inputs.shape.as_list()
        # x, y = tf.meshgrid(tf.range(w), tf.range(h))
        # w_loc = slim.one_hot_encoding(x, num_classes=w)
        # h_loc = slim.one_hot_encoding(y, num_classes=h)
        # loc = tf.concat([h_loc, w_loc], 2)
        # loc = tf.tile(tf.expand_dims(loc, 0), [BATCH_SIZE, 1, 1, 1])
        # return tf.concat([inputs, loc], 3)
        embedd_size = 64
        shape = tf.shape(inputs)
        batch_size, h, w = shape[0],shape[1],shape[2]
        image_width = inputs.get_shape().dims[2].value
        x = tf.range(w*h)
        x = tf.reshape(x, [1, h, w, 1])
        loc = tf.tile(x, [batch_size, 1, 1, 1])
        embedding = tf.get_variable("embedding", initializer=tf.random_uniform([image_width, embedd_size], -1.0, 1.0)) 
        loc = tf.nn.embedding_lookup(embedding, loc)
        loc = tf.squeeze(loc, squeeze_dims=3)
        loc = tf.concat([inputs, loc], 3)
        return loc


# 将CNN模型转换到SEQ序列
def CNN2SEQ(inputs):
    with tf.variable_scope("CNN2SEQ"):
        # batch_size = inputs.get_shape().dims[0].value
        w = inputs.get_shape().dims[1].value
        h = inputs.get_shape().dims[2].value
        feature_size = inputs.get_shape().dims[3].value
        return tf.reshape(inputs, [-1, w*h, feature_size])

# 采用标准正交基的方式初始化参数
# 给LSTM使用，可以提升初始化效果
def orthogonal_initializer(shape, dtype=tf.float32, *args, **kwargs):
    del args
    del kwargs
    # 扁平化shape
    flat_shape = (shape[0], np.prod(shape[1:]))
    # 标准正态分布
    w = np.random.randn(*flat_shape)
    # 奇异值分解，提取矩阵特征，返回的 u,v 都是正交随机的，
    # u,v 分别代表了batch之间的数据相关性和同一批数据之间的相关性，_ 值代表了批次和数据的交叉相关性，舍弃掉
    u, _, v = np.linalg.svd(w, full_matrices=False)
    w = u if u.shape == flat_shape else v
    return tf.constant(w.reshape(shape), dtype=dtype)

# 注意力模型
# 输入[B T F]
def Attention(net, labels_one_hot):
    with tf.variable_scope("Attention"):
        regularizer = slim.l2_regularizer(0.00004)
        _softmax_w = slim.model_variable('softmax_w', [LSTM_UNITS_NUMBER, CLASSES_NUMBER],
            initializer=orthogonal_initializer, regularizer=regularizer)
        _softmax_b = slim.model_variable('softmax_b', [CLASSES_NUMBER],
            initializer=tf.zeros_initializer(), regularizer=regularizer)
        _zero_label = tf.zeros([BATCH_SIZE, CLASSES_NUMBER])

        first_label = _zero_label
        decoder_inputs = [first_label] + [None] * (SEQ_LENGTH - 1)
        lstm_cell = tf.contrib.rnn.LSTMCell(LSTM_UNITS_NUMBER, use_peepholes=False, 
            cell_clip=10., state_is_tuple=True, initializer=orthogonal_initializer)

        _char_logits={}
        def char_logit(inputs, char_index):
            if char_index not in _char_logits:
                _char_logits[char_index] = tf.nn.xw_plus_b(inputs, _softmax_w, _softmax_b)
            return _char_logits[char_index]

        def char_one_hot(self, logit):
            prediction = tf.argmax(logit, axis=1)
            return slim.one_hot_encoding(prediction, CLASSES_NUMBER)
            
        def get_input(prev, i):
            if i==0:
                return _zero_label
            else:
                if labels_one_hot!=None:
                    return labels_one_hot[:, i - 1, :]
                else:
                    logit = char_logit(prev, char_index=i - 1)
                    return char_one_hot(logit)

        lstm_outputs, _ = tf.contrib.legacy_seq2seq.attention_decoder(
            decoder_inputs=decoder_inputs,
            initial_state=lstm_cell.zero_state(BATCH_SIZE, tf.float32),
            attention_states=net,
            cell=lstm_cell,
            loop_function=get_input)
    
        logits_list = [
            tf.expand_dims(char_logit(logit, i), dim=1)
            for i, logit in enumerate(lstm_outputs)
        ]
        
        return tf.concat(logits_list, 1)

def logits_to_log_prob(logits):
    with tf.variable_scope('log_probabilities'):
        # 将所有的值都降到0和以下
        reduction_indices = len(logits.shape.as_list()) - 1
        # 取最大值 max_logits 
        max_logits = tf.reduce_max(logits, reduction_indices=reduction_indices, keep_dims=True)
        
        # 都降到 0 以下 
        safe_logits = tf.subtract(logits, max_logits)
        # exp(-x) => (0 ~ 1) 求和最后一个维度
        sum_exp = tf.reduce_sum(tf.exp(safe_logits),  reduction_indices=reduction_indices, keep_dims=True)
        # 再将 log(sum) => (0 ~ 1)  
        log_probs = tf.subtract(safe_logits, tf.log(sum_exp))  
        return log_probs

# 文字预测
def char_predictions(chars_logit):
    # 稳定网络
    log_prob = logits_to_log_prob(chars_logit)

    # 取出最大概率的编号
    ids = tf.to_int32(tf.argmax(log_prob, axis=2), name='predicted_chars')

    # onehot ids
    mask = tf.cast(slim.one_hot_encoding(ids, CLASSES_NUMBER), tf.bool)
    
    # 将概率统一到1以内
    all_scores = tf.nn.softmax(chars_logit)

    # 只取出 ids 的概率，转为 [batch_size, seq_length]
    selected_scores = tf.boolean_mask(all_scores, mask, name='char_scores')
    scores = tf.reshape(selected_scores, shape=(-1, SEQ_LENGTH))
    return ids, log_prob, scores

# 损失函数
# 这里删除了 label_smoothing ， 后续有时间测试下差异
def sequence_loss_fn(chars_logits, chars_labels):    
    with tf.variable_scope('sequence_loss_fn'):        
        labels_list = tf.unstack(chars_labels, axis=1)
        batch_size, seq_length, _ = chars_logits.shape.as_list()
        weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
        logits_list = tf.unstack(chars_logits, axis=1)
        weights_list = tf.unstack(weights, axis=1)
        loss = tf.contrib.legacy_seq2seq.sequence_loss(
            logits_list,
            labels_list,
            weights_list,
            softmax_loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits,
            average_across_timesteps = False)
        return loss

def OCR(inputs, labels_one_hot, labels, reuse = False):
    with tf.variable_scope("OCR", reuse=reuse):
        layer = inputs
        print("Image shape:",layer.shape)

        # CNN
        layer = CNN(layer)
        print("CNN shape:",layer.shape)

        # Coordinates
        layer = Coordinates(layer)
        print("Coordinates shape:", layer.shape)    

        layer = CNN2SEQ(layer)
        print("CNN2SEQ shape:", layer.shape)

        # 文字模型
        chars_logit = Attention(layer, labels_one_hot)    
        print('chars_logit:', chars_logit.shape)

        # 预测的字符， 稳定后的文字模型， 预测的概率
        predicted_chars, chars_log_prob, predicted_scores = (char_predictions(chars_logit))

        # create_loss
        chars_loss = sequence_loss_fn(chars_logit, labels)
        tf.summary.scalar('TotalLoss', chars_loss)
        return chars_logit, chars_log_prob, predicted_chars, predicted_scores, chars_loss

def create_optimizer(optimizer_name):
    if optimizer_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(LEARNING_RATE_INITIAL, momentum=0.9)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL)
    elif optimizer_name == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE_INITIAL)
    elif optimizer_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(LEARNING_RATE_INITIAL)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE_INITIAL, momentum=0.9)
    return optimizer

def accuracy(predictions, targets):
    accuracy_values=[]
    with tf.variable_scope('CharAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())
        targets = tf.to_int32(targets)
        const_rej_char = tf.constant(NULL_CODE, shape=targets.get_shape(), dtype=tf.int32)
        weights = tf.to_float(tf.not_equal(targets, const_rej_char))
        correct_chars = tf.to_float(tf.equal(predictions, targets))
        accuracy_per_example = tf.div(
            tf.reduce_sum(tf.multiply(correct_chars, weights), 1),
            tf.reduce_sum(weights, 1))
        accuracy_value = tf.contrib.metrics.streaming_mean(accuracy_per_example)
        accuracy_values.append(accuracy_value)
        tf.summary.scalar('CharAccuracy', tf.Print(accuracy_value, [accuracy_value], 'CharAccuracy'))

    with tf.variable_scope('SequenceAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())
        targets = tf.to_int32(targets)
        const_rej_char = tf.constant(NULL_CODE, shape=targets.get_shape(), dtype=tf.int32)
        include_mask = tf.not_equal(targets, const_rej_char)
        include_predictions = tf.to_int32(tf.where(include_mask, predictions, tf.zeros_like(predictions) + NULL_CODE))
        correct_chars = tf.to_float(tf.equal(include_predictions, targets))
        correct_chars_counts = tf.cast(tf.reduce_sum(correct_chars, reduction_indices=[1]), dtype=tf.int32)
        target_length = targets.get_shape().dims[1].value
        target_chars_counts = tf.constant(target_length, shape=correct_chars_counts.get_shape())
        accuracy_per_example = tf.to_float(tf.equal(correct_chars_counts, target_chars_counts))
        accuracy_value = tf.contrib.metrics.streaming_mean(accuracy_per_example)
        accuracy_values.append(accuracy_value)
        tf.summary.scalar('SequenceAccuracy', tf.Print(accuracy_value, [accuracy_value], 'SequenceAccuracy'))

    return accuracy_values

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="inputs")
    labels = tf.placeholder(tf.int32,[None, SEQ_LENGTH], name="labels")
    labels_onehot = slim.one_hot_encoding(labels, CLASSES_NUMBER)

    global_step = tf.Variable(0, trainable=False)
    lr = tf.Variable(LEARNING_RATE_INITIAL, trainable=False)

    chars_logit, chars_log_prob, predicted_chars, predicted_scores, chars_loss = OCR(inputs, labels_onehot, labels)
    ocr_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='OCR')

    # ocr_optim = tf.train.AdamOptimizer(lr).minimize(chars_loss, global_step=global_step)
    ocr_optim = create_optimizer("momentum").minimize(chars_loss, global_step=global_step) 

    oc_accs = accuracy(predicted_chars, labels)

    # 加入日志
    tf.summary.scalar('ocr_loss', chars_loss)
    # res_images = res_layer[-1]
    # res_images = tf.transpose(res_images, perm=[2, 0, 1])
    # tf.summary.image('net_res', tf.expand_dims(res_images,-1), max_outputs=9)
    for var in ocr_vars:
        tf.summary.histogram(var.name, var)

    summary = tf.summary.merge_all()

    return  inputs, labels, global_step, lr, summary, \
            chars_loss, ocr_optim, oc_accs[0], oc_accs[1]

def list_to_chars(list):
    try:
        return "".join([CHARS[v] for v in list])
    except Exception as err:
        return "Error: %s" % err        



# get_next_batch_for_res(1)

def train():
    inputs, labels, global_step, lr, summary, \
        chars_loss, ocr_optim, cacc, sacc  = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_R_dir = os.path.join(model_dir, "RL32")
    if not os.path.exists(model_R_dir): os.mkdir(model_R_dir)

    log_dir = os.path.join(model_dir, "logs")
    if not os.path.exists(log_dir): os.mkdir(log_dir)

    with tf.Session() as session:
        print("tf init")
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # session.run(init_op)
        session.run(tf.global_variables_initializer())

        print("tf check restore")
        # r_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='OCR'), sharded=True, max_to_keep=5)
        r_saver = tf.train.Saver(max_to_keep=5)

        for i in range(3):
            ckpt = tf.train.get_checkpoint_state(model_R_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restore Model OCR...")
                stem = os.path.basename(ckpt.model_checkpoint_path)
                restore_iter = int(stem.split('-')[-1])
                try:
                    r_saver.restore(session, ckpt.model_checkpoint_path)    
                except:
                    new_restore_iter = restore_iter - BATCHES
                    with open(os.path.join(model_R_dir,"checkpoint"),'w') as f:
                        f.write('model_checkpoint_path: "OCR.ckpt-%s"\n'%new_restore_iter)
                        f.write('all_model_checkpoint_paths: "OCR.ckpt-%s"\n'%new_restore_iter)
                    continue
                session.run(tf.assign(global_step, restore_iter))
                if restore_iter<10000:        
                    session.run(tf.assign(lr, 1e-4))
                elif restore_iter<50000:            
                    session.run(tf.assign(lr, 1e-5))
                else:
                    session.run(tf.assign(lr, 1e-6))
                print("Restored to %s."%restore_iter)
                break
            else:
                break
            print("restored fail, return")
            return

        print("tf create summary")
        train_writer = tf.summary.FileWriter(log_dir, session.graph)

        print("tf train")

        AllLosts={}
        accs = deque(maxlen=100)
        losts = deque(maxlen=200)
        while True:
            errR = 1
            batch_size = BATCH_SIZE
            for batch in range(BATCHES):
                start = time.time()    
                train_inputs, train_labels, _, _, train_info =  font_dataset.get_next_batch_for_res(batch_size,
                    has_sparse=False, has_onehot=False, max_width=4096, height=32, need_pad_width_to_max_width=True)
                train_labels_fix = np.ones((batch_size, SEQ_LENGTH))
                train_labels_fix *= (NULL_CODE)
                for i in range(batch_size):
                    np.put(train_labels_fix[i],np.arange(len(train_labels[i])),train_labels[i])

                feed = {inputs: train_inputs, labels: train_labels_fix} 

                feed_time = time.time() - start
                start = time.time()    

                # _res = session.run(net_res, feed)
                # print(_res.shape)

                errR, _ , steps, res_lr, char_acc, seq_acc  = session.run([chars_loss, ocr_optim, global_step, lr, cacc, sacc], feed)

                font_length = int(train_info[0][-1])
                font_info = train_info[0][0]+"/"+train_info[0][1]+"/"+str(font_length)
                # accs.append(acc)
                # avg_acc = sum(accs)/len(accs)

                losts.append(errR)
                avg_losts = sum(losts)/len(losts)

                # errR = errR / font_length
                print("%s, %d time: %4.4fs / %4.4fs, loss: %.4f, avg_loss: %.4f, acc: %.8f / %.8f,  lr:%.8f, info: %s " % \
                    (time.ctime(), steps, feed_time, time.time() - start, errR, avg_losts, char_acc, seq_acc, res_lr, font_info))

                # 如果当前lost低于平均lost，就多训练
                # need_reset_global_step = False
                # for _ in range(10):
                #     if errR <=  avg_losts*2: break 
                #     start = time.time()                
                #     errR, acc, _, res_lr = session.run([res_loss, res_acc, res_optim, lr], feed)
                #     accs.append(acc)
                #     avg_acc = sum(accs)/len(accs)                  
                #     print("%s, %d time: 0.0000s / %4.4fs, acc: %.4f, avg_acc: %.4f, loss: %.4f, avg_loss: %.4f, lr:%.8f, info: %s " % \
                #         (time.ctime(), steps, time.time() - start, acc, avg_acc, errR, avg_losts, res_lr, font_info))   
                #     need_reset_global_step = True
                    
                # if need_reset_global_step:                     
                #     session.run(tf.assign(global_step, steps))

                # if np.isnan(errR) or np.isinf(errR) :
                #     print("Error: cost is nan or inf")
                #     return

                # for info in train_info:
                #     key = ",".join(info)
                #     if key in AllLosts:
                #         AllLosts[key]=AllLosts[key]*0.99+acc*0.01
                #     else:
                #         AllLosts[key]=acc

                # if acc/avg_acc<=0.2:
                #     for i in range(batch_size): 
                #         filename = "%s_%s_%s_%s_%s_%s_%s.png"%(acc, steps, i, \
                #             train_info[i][0], train_info[i][1], train_info[i][2], train_info[i][3])
                #         cv2.imwrite(os.path.join(curr_dir,"test",filename), train_inputs[i] * 255)                    
                # 报告
                # if steps >0 and steps % REPORT_STEPS == 0:
                #     train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size)   
           
                #     decoded_list = session.run(res_decoded[0], {inputs: train_inputs, seq_len: train_seq_len}) 

                #     for i in range(batch_size): 
                #         cv2.imwrite(os.path.join(curr_dir,"test","%s_%s.png"%(steps,i)), train_inputs[i] * 255) 

                #     original_list = utils.decode_sparse_tensor(train_labels)
                #     detected_list = utils.decode_sparse_tensor(decoded_list)
                #     if len(original_list) != len(detected_list):
                #         print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                #             " test and detect length desn't match")
                #     print("T/F: original(length) <-------> detectcted(length)")
                #     acc = 0.
                #     for idx in range(min(len(original_list),len(detected_list))):
                #         number = original_list[idx]
                #         detect_number = detected_list[idx]  
                #         hit = (number == detect_number)
                #         print("----------",hit,"------------")          
                #         print(list_to_chars(number), "(", len(number), ")")
                #         print(list_to_chars(detect_number), "(", len(detect_number), ")")
                #         # 计算莱文斯坦比
                #         import Levenshtein
                #         acc += Levenshtein.ratio(list_to_chars(number),list_to_chars(detect_number))
                #     print("Test Accuracy:", acc / len(original_list))
                #     sorted_fonts = sorted(AllLosts.items(), key=operator.itemgetter(1), reverse=False)
                #     for f in sorted_fonts[:20]:
                #         print(f)

                    # if avg_losts>100:        
                    #     session.run(tf.assign(lr, 1e-4))
                    # elif avg_losts>10:            
                    #     session.run(tf.assign(lr, 1e-5))
                    # else:
                    #     session.run(tf.assign(lr, 1e-6))
                                            
            # 如果当前 loss 为 nan，就先不要保存这个模型
            if np.isnan(errR) or np.isinf(errR):
                continue
            print("Save Model OCR ...")
            r_saver.save(session, os.path.join(model_R_dir, "OCR.ckpt"), global_step=steps)         
            logs = session.run(summary, feed)
            train_writer.add_summary(logs, steps)

if __name__ == '__main__':
    train()