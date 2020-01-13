import _Base_ as base
import numpy as np
import torch
from torch.autograd import Variable


def test(model, loader):
    '''进行一轮测试，直接输出结果'''
    from torch.autograd import Variable
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)

    for x, y in loader:
        x_var = Variable(x, True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
    acc = float(num_correct) / float(num_samples) * 100
    return num_correct, num_samples, acc


def pred_batch(x, model):
    '''批量返回预测结果'''
    x = Variable(x, True)
    scores = model(x)
    _, preds = scores.data.cpu().max(1)
    return preds
