import Models.Lenet5
import Models.My_Model

def load_state_dict(net,filePath):
    import os
    import torch
    if os.path.exists(filePath):#加载模型的参数
        net.load_state_dict(torch.load(filePath))
    return net