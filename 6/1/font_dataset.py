# coding=utf-8
###############
# 本程序用于读取预先创建好的font文件
# 和 create_font_dataset 配合使用
###############
import tensorflow as tf
import os
import utils_pil, utils_font, utils

image_height = 32
# image_size = 512
# resize_image_size = 256
# 所有 unicode CJK统一汉字（4E00-9FBB） + ascii的字符加 + ctc blank
# https://zh.wikipedia.org/wiki/Unicode
# https://zh.wikipedia.org/wiki/ASCII
ASCII_CHARS = [chr(c) for c in range(32,126+1)]
#ZH_CHARS = [chr(c) for c in range(int('4E00',16),int('9FBB',16)+1)]
#ZH_CHARS_PUN = ['。','？','！','，','、','；','：','「','」','『','』','‘','’','“','”',\
#                '（','）','〔','〕','【','】','—','…','–','．','《','》','〈','〉']
CHARS = ASCII_CHARS #+ ZH_CHARS + ZH_CHARS_PUN
# CHARS = ASCII_CHARS
# 将空格移动到第一个
CHARS.remove(" ")
CHARS.insert(0, " ")
CLASSES_NUMBER = len(CHARS) + 1 
MAX_IMAGE_WIDTH = 4096
MAX_SEQ_LENGTH = 1024

curr_dir = os.path.dirname(__file__)

def dataset_init():
    data_dir = os.path.join(curr_dir,"data")
    datafiles = os.listdir(data_dir)
    data_file = os.path.join(data_dir, random.choice(datafiles))
    print("load data_file", data_file)
    return tf.python_io.tf_record_iterator(data_file)

dataset = dataset_init()
dataset_example=tf.train.Example() 

def get_next_batch_for_res(batch_size=128, is_sparse=True):
    inputs_images = []   
    codes = []
    max_width_image = 0
    info = []
    seq_len = np.ones(batch_size)

    for i in range(batch_size):
        serialized_example = next(dataset, None)
        if serialized_example==None:
            raise Exception("has finished train one data file, stop")

        dataset_example.ParseFromString(serialized_example)

        font_name = str(dataset_example.features.feature['font_name'].bytes_list.value[0],  encoding="utf-8")
        font_size = dataset_example.features.feature['font_size'].int64_list.value[0]
        font_mode = dataset_example.features.feature['font_mode'].int64_list.value[0]
        font_hint = dataset_example.features.feature['font_mode'].int64_list.value[0]

        text = str(dataset_example.features.feature['label'].bytes_list.value[0],  encoding="utf-8")
        size = dataset_example.features.feature['size'].int64_list.value
        image = dataset_example.features.feature['image'].bytes_list.value[0]
        image = utils_pil.frombytes(tuple(size), image)

        image = utils_pil.convert_to_gray(image) 
        w, h = size
        if h > image_height:
            image = utils_pil.resize_by_height(image, image_height)  

        image = utils_pil.resize_by_height(image, image_height-random.randint(1,5))
        image, _ = utils_pil.random_space2(image, image,  image_height)
        
        image = utils_font.add_noise(image)   
        image = np.asarray(image) 

        image = utils.resize(image, image_height, MAX_IMAGE_WIDTH)

        if random.random()>0.5:
            image = image / 255.
        else:
            image = (255. - image) / 255.

        if max_width_image < image.shape[1]:
            max_width_image = image.shape[1]
          
        inputs_images.append(image)

        codes.append([CHARS.index(char) for char in text])

        info.append([font_name, str(font_size), str(font_mode), str(font_hint), str(len(text))])

    # 凑成4的整数倍
    if max_width_image % 4 > 0:
        max_width_image = max_width_image + 4 - max_width_image % 4

    # 如果图片超过最大宽度
    if max_width_image > MAX_IMAGE_WIDTH:
        raise Exception("img width must %s <= %s " % (max_width_image, MAX_IMAGE_WIDTH))

    inputs = np.zeros([batch_size, image_height, max_width_image, 1])
    for i in range(batch_size):
        image_vec = utils.img2vec(inputs_images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.reshape(image_vec,(image_height, max_width_image, 1))
     

    labels = [np.asarray(i) for i in codes]

    out_labels = []
    if is_sparse:
        out_labels =  utils.sparse_tuple_from(labels)
    else:
        for label in labels:
            label_one_hot = np.zeros((len(label), CLASSES_NUMBER))  
            label_one_hot[:, label] = 1
            out_labels.append(label_one_hot)
            # out_labels.append(slim.one_hot_encoding(label, CLASSES_NUMBER))

    out_labels = np.array(out_labels)
    seq_len = np.ones(batch_size) * (max_width_image//4)
    # print(inputs.shape, seq_len.shape, [len(l) for l in labels])
    return inputs, out_labels, seq_len, info

def get_next_batch_for_res_onehot(batch_size=128):
    return get_next_batch_for_res(False)

def get_next_batch_for_res_sparse(batch_size=128):
    return get_next_batch_for_res(True)
