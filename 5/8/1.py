# coding=utf-8
#!/sbin/python

import numpy as np
import paddle.v2 as paddle
import json
import os
import random
import sys

curr_dir = os.path.dirname(__file__)

data_path = os.path.join(curr_dir,"data")


class_dim = 3 # 0 不是关键 1 是关键 2 重复关键

def load_data():
    data = json.loads(open(os.path.join(data_path,"meta.json")).read())
    training_data = []
    validation_data = []
    testing_data = []
    for data_id in data['database']:
        if data['database'][data_id]['subset'] == 'training':
            if os.path.exists(os.path.join(data_path,"training", "%s.pkl"%data_id)):
                training_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
        elif data['database'][data_id]['subset'] == 'validation':
            if os.path.exists(os.path.join(data_path,"validation", "%s.pkl"%data_id)):
                validation_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
        elif data['database'][data_id]['subset'] == 'testing':
            if os.path.exists(os.path.join(data_path,"testing", "%s.pkl"%data_id)):
                testing_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
    print('load data train %s, valid %s, test %s'%(len(training_data), len(validation_data), len(testing_data)))
    return training_data, validation_data, testing_data

# def load_data():
#     data = json.loads(open(os.path.join(data_path,"meta.json")).read())
#     training_data = []
#     validation_data = []
#     testing_data = []
#     _training_data = json.loads(open(os.path.join(data_path,"training","data.json")).read())
#     _validation_data = json.loads(open(os.path.join(data_path,"validation","data.json")).read())
#     _testing_data = json.loads(open(os.path.join(data_path,"testing","data.json")).read())
#     for data_id in data['database']:
#         f = "%s.pkl"%data_id
#         if data['database'][data_id]['subset'] == 'training':
#             if f in _training_data:
#                 training_data.append({'id':data_id,'data':data['database'][data_id]['annotations'],"shape":_training_data[f]})
#         elif data['database'][data_id]['subset'] == 'validation':
#             if f in _validation_data:
#                 validation_data.append({'id':data_id,'data':data['database'][data_id]['annotations'],"shape":_validation_data[f]})
#         elif data['database'][data_id]['subset'] == 'testing':
#             if f in _testing_data:
#                 testing_data.append({'id':data_id,'data':data['database'][data_id]['annotations'],"shape":_testing_data[f]})
#     print('load data train %s, valid %s, test %s'%(len(training_data), len(validation_data), len(testing_data)))
#     return training_data, validation_data, testing_data


def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(), ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, act=active_type)

def shortcut(ipt, n_in, n_out, stride):
    if n_in != n_out:
        return conv_bn_layer(ipt, n_out, 1, stride, 0, paddle.activation.Linear())
    else:
        return ipt

def basicblock(ipt, ch_out, stride):
    ch_in = ch_out * 2
    tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Relu())

def layer_warp(block_func, ipt, features, count, stride):
    tmp = block_func(ipt, features, stride)
    for i in range(1, count):
        tmp = block_func(tmp, features, 1)
    return tmp

def resnet_cifar10(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(ipt, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 2)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(input=res3, pool_size=9, stride=1, pool_type=paddle.pooling.Avg())
    return pool

def network():
    # -1 ,2048 
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2048*3))
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(3))

    layer = paddle.layer.fc(input=x, size=72*72*3, act=paddle.activation.Relu())
    layer = resnet_cifar10(layer)
    
    output = paddle.layer.fc(input=layer, size=class_dim, act=paddle.activation.Softmax())
    cost = paddle.layer.classification_cost(input=output, label=y)
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=2e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))
    return cost, parameters, adam_optimizer

training_data, validation_data, testing_data = load_data()  
def reader_get_image_and_label():
    def reader():
        for data in training_data:
            # data = random.choice(training_data)
            batch_data = np.zeros((2048,3))            
            v_data = np.load(os.path.join(data_path,"training", "%s.pkl"%data["id"]))
            w = v_data.shape[0]
            label = np.zeros([w], dtype=np.int)

            for annotations in data["data"]:
                segment = annotations['segment']
                for i in range(int(segment[0]),int(segment[1]+1)):
                    label[i] += 1

            for i in range(len(label)):
                _data = np.reshape(v_data[i], (2048,1))
                np.append(batch_data[:, 1:], _data, axis=1)
                yield batch_data, label[i]
    return reader

# training_data, validation_data, testing_data = load_data()  
# def reader_get_image_and_label():
#     def reader():
#         for data in training_data:
#             batch_data = np.zeros((2048,3))
#             print("reading:", data["id"], " shape:", data["shape"])
#             v_data = np.random.random(data["shape"])
#             w = v_data.shape[0]
#             label = np.zeros([w], dtype=np.int)
#             for annotations in data["data"]:
#                 segment = annotations['segment']
#                 for i in range(int(segment[0]),int(segment[1]+1)):
#                     label[i] += 1
#             for i in range(len(label)):
#                 _data = np.reshape(v_data[i], (2048,1))
#                 np.append(batch_data[:, 1:], _data, axis=1)
#                 yield batch_data, label[i]
#     return reader

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print("\nPass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics) )
            with open('parameters.tar', 'w') as f:
                print("saveing parameters ...")
                trainer.save_parameter_to_tar(f)    
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
            trainer.save_parameter_to_tar(f)

#         result = trainer.test(
#             reader=paddle.batch(
#                 paddle.dataset.cifar.test10(), batch_size=128),
#             feeding=feeding)
#         print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
        
def train():
    print("paddle init ...")
    paddle.init(use_gpu=False, trainer_count=1)  
    print("get network ...")
    cost, parameters, adam_optimizer = network()
    if os.path.exists("parameters.tar"):
        print("loading parameters ...")
        parameters.from_tar(open("parameters.tar"))       
    print('set reader ...')
    train_reader = paddle.batch(reader_get_image_and_label(), batch_size=10)
    feeding={'x': 0, 'y': 1}
    trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=adam_optimizer)
    print("start train ...")
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=1)

if __name__ == '__main__':
    train()