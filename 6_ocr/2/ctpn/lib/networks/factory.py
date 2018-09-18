from .VGGnet_test import VGGnet_test
from .VGGnet_train import VGGnet_train

# 只提供了训练和测试的VGG的网络，差别就是取消了BOX的部分输入
def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
           return VGGnet_test()
        elif name.split('_')[1] == 'train':
           return VGGnet_train()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))
