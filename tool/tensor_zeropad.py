import numpy as np
import torch.nn.functional as F


def zeropad_to_size(small_tensor, big_tensor_size):
    """
    将一个小向量进行0填充扩展,可用于模型聚合时,将小模型的tensor进行扩展
    :param small_tensor: 需要0填充的向量
    :param big_tensor_size: 需要0填充到的形状
    :return: 0填充到big_tensor_size大小的small_tensor
    """
    diff = np.array(big_tensor_size) - np.array(small_tensor.size())
    #  if 'diff' is: np.array([20,10,0,0])
    #  then pad is: (0,0,0,0,0,10,0,20)
    pad = []
    for x in diff[::-1]:
        if x == 0:
            pad.append(0)
            pad.append(0)
        else:
            pad.append(0)
            pad.append(x)
    pad = tuple(pad)
    return F.pad(small_tensor, pad, 'constant', 0)
