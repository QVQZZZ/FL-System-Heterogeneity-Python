import numpy as np
import torch.nn.functional as F


def cut_to_size(big_tensor, small_tensor_size):
    """
    将一个大向量进行裁剪,可用于客户端从服务器获取模型时,对模型进行裁剪
    :param big_tensor: 需要裁剪的向量
    :param small_tensor_size: 裁剪到什么尺寸
    :return: 裁剪到small_tensor_size大小的big_tensor
    """
    diff = np.array(small_tensor_size) - np.array(big_tensor.size())
    #  if 'diff' is: np.array([-10,-5,0,0])
    #  then pad is: (0,0,0,0,0,-5,0,-10)
    pad = []
    for x in diff[::-1]:
        if x == 0:
            pad.append(0)
            pad.append(0)
        else:
            pad.append(0)
            pad.append(x)
    pad = tuple(pad)
    return F.pad(big_tensor, pad, 'constant', 0)
