import os
import tensorflow as tf
import numpy as np
import ocr
import utils
import cv2
import random

curr_dir = os.path.dirname(__file__)

def init():
    logits, inputs, labels, seq_len, W, b, input_keep_prob = ocr.neural_networks()
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # restore sess
    model_dir = os.path.join(curr_dir, "model")
    saver_prefix = os.path.join(model_dir, "model.ckpt")        
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver(max_to_keep=5)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restore Model ...")
        saver.restore(sess, ckpt.model_checkpoint_path)    
    return sess, inputs, seq_len, input_keep_prob, decoded, log_prob



def scan():
    # session, inputs, seq_len, input_keep_prob, decoded, log_prob = init()
    need_ocr_images = utils.loadImage(os.path.join(curr_dir,'test','0.jpg'))
    ocr_text_groups = []
    for idx,images_group in enumerate(need_ocr_images):
        # if idx != 1: continue
        ocr_texts=[]

        ocr_inputs = np.zeros([len(images_group), ocr.image_size[1], ocr.image_size[0]])
        for i, image in enumerate(images_group):
            print(i,image.shape)
            image_vec = utils.img2vec(image,ocr.image_size[0],ocr.image_size[1])
            ocr_inputs[i,:] = np.transpose(image_vec.reshape((ocr.image_size[0],ocr.image_size[1])))           
            # utils.show(image)
            
        ocr_seq_len = np.ones(ocr_inputs.shape[0]) * ocr.image_size[1]

        feed = {inputs: ocr_inputs, seq_len: ocr_seq_len,  input_keep_prob: 1.0}
        print("starting ocr inputs...")
        decoded_list,_  = session.run([decoded[0],log_prob], feed)
        print("filished ocr inputs...")
        detected_list = ocr.decode_sparse_tensor(decoded_list)
        for detect_number in detected_list:
            ocr_texts.append(ocr.list_to_chars(detect_number))

        ocr_text_groups.append(ocr_texts)   
    return ocr_text_groups             

if __name__ == '__main__':
    # img = cv2.imread(os.path.join(curr_dir,'test','1.png'), 0)
    # print(img.shape)
    # split_images = utils.splitImg(img)
    # print(split_images[0].shape)
    ocr_text = scan()
    #print(ocr_text)




