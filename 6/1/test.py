import os
import tensorflow as tf
import numpy as np
import utils
import cv2
import random
import time
import ocr

curr_dir = os.path.dirname(__file__)

def init():
#    logits, inputs, labels, seq_len, input_keep_prob = ocr.neural_networks()
    global_step = tf.Variable(0, trainable=False)
    curr_learning_rate = 1e-5
    learning_rate = tf.placeholder(tf.float32, shape=[])                                            
    logits, inputs, labels, seq_len, input_keep_prob = ocr.neural_networks()
    loss = tf.nn.ctc_loss(labels=labels,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss, name="cost")
    grads_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = grads_optimizer.compute_gradients(loss)
    capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
    optimizer = grads_optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=10, merge_repeated=False)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # restore sess
    model_dir = os.path.join(curr_dir, "model")
    saver_prefix = os.path.join(model_dir, "model.ckpt")        
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        print("Restore Model ...")
        saver.restore(sess, ckpt.model_checkpoint_path)    
    return sess, inputs, seq_len, input_keep_prob, decoded, log_prob



def scan():
    session, inputs, seq_len, input_keep_prob, decoded, log_prob = init()
    need_ocr_images = utils.loadImage(os.path.join(curr_dir,'test','1.jpg'),0)
    ocr_text_groups = []
    for idx,images_group in enumerate(need_ocr_images):
        # if idx != 1: continue
        ocr_texts=[]

        # 取最大宽度做为本组的统一输入长度
        widths=[image.shape[1] for image in images_group]
        maxImageWidth = max(widths)+5

        ocr_inputs = np.zeros([len(images_group), maxImageWidth, ocr.image_height])
        for i, image in enumerate(images_group):
            # image = utils.dropZeroEdges(image)
            # utils.show(utils.dropZeroEdges(image))
            utils.save(image,os.path.join(curr_dir,"test","%s-%s.png"%(idx,i)))
            image_vec = utils.img2vec(image,ocr.image_height,maxImageWidth)
            ocr_inputs[i,:] = np.transpose(image_vec.reshape((ocr.image_height,maxImageWidth)))         
            # utils.show(image)
            # return
        ocr_seq_len = np.ones(ocr_inputs.shape[0]) * maxImageWidth

        feed = {inputs: ocr_inputs, seq_len: ocr_seq_len,  input_keep_prob: 1.0}
        print("starting ocr inputs %s ..." % idx)
        start = time.time()
        decoded_list = session.run(decoded[0], feed)
        seconds = round(time.time() - start,2)
        print("filished ocr inputs %s, paid %s seconds" % (idx,seconds))
        detected_list = ocr.decode_sparse_tensor(decoded_list)
        for detect_number in detected_list:
            ocr_texts.append(ocr.list_to_chars(detect_number))

        ocr_text_groups.append(ocr_texts)
        # break   
    return ocr_text_groups             

def scan2():
    session, inputs, seq_len, input_keep_prob, decoded, log_prob = init()
    need_ocr_images = utils.loadImage(os.path.join(curr_dir,'test','0.jpg'),0)
    ocr_text_groups = []
    for idx,images_group in enumerate(need_ocr_images):
        # if idx != 1: continue
        ocr_texts=[]

        for i, image in enumerate(images_group):
            image = utils.dropZeroEdges(image)  
            image = utils.resize(image,ocr.image_height)
            utils.save(image,os.path.join(curr_dir,"test","%s-%s.png"%(idx,i)))
            maxImageWidth = image.shape[1]+5
            image_vec = utils.img2vec(image,ocr.image_height,maxImageWidth)
            ocr_inputs = np.zeros([1, maxImageWidth, ocr.image_height])
            ocr_inputs[0,:] = np.transpose(image_vec.reshape((ocr.image_height,maxImageWidth)))         
            ocr_seq_len = np.ones(ocr_inputs.shape[0]) * maxImageWidth
            feed = {inputs: ocr_inputs, seq_len: ocr_seq_len,  input_keep_prob: 1.0}
            print("starting ocr inputs %s:%s ..." % (idx,i))
            start = time.time()
            decoded_list = session.run(decoded[0], feed)
            seconds = round(time.time() - start,2)
            print("filished ocr inputs %s, paid %s seconds" % (idx,seconds))
            detected_list = ocr.decode_sparse_tensor(decoded_list)            
            for detect_number in detected_list:
                ocr_texts.append(ocr.list_to_chars(detect_number))

        ocr_text_groups.append(ocr_texts)
        # break   
    return ocr_text_groups  


if __name__ == '__main__':
    # img = cv2.imread(os.path.join(curr_dir,'test','1.png'), 0)
    # print(img.shape)
    # split_images = utils.splitImg(img)
    # print(split_images[0].shape)
    ocr_text = scan2()
    print(ocr_text)




