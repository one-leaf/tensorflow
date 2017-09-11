import os
import tensorflow as tf


def init():
    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, "model")

    checkpoint_prefix = os.path.join(model_dir, "model.ckpt")

    # 找到最新的运算模型文件
    metaFile= sorted(
        [
            (x, os.path.getctime(os.path.join(model_dir,x)))                  
            for x in os.listdir(model_dir) if x.endswith('.meta')  
        ],
        key=lambda i: i[1])[-1][0]

    sess = tf.Session()

    saver = tf.train.import_meta_graph(os.path.join(model_dir,metaFile))
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception("error: can't load checkpoint data")
    # for tensor in tf.get_default_graph().as_graph_def().node:
    #     print(tensor.name,tensor.attr['shape'])
    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    decoded = tf.get_default_graph().get_tensor_by_name('decoded:0')
    return sess,inputs,decoded

sess,inputs,decoded = init()


