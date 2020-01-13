import torch


def enable_cuda(obj):
    '''使用cuda加速计算'''
    import torch
    if torch.cuda.is_available():  # 成功
        obj.cuda()
        return True
    else:  # 失败
        return False


def to_var(self, requires_grad=False):
    '''变量格式转换为Variable'''
    if torch.cuda.is_available():
        self = self.cuda()
    with torch.no_grad():
        result = torch.autograd.Variable(self, requires_grad=requires_grad)
    return result
