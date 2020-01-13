import torch
import Data_Reader.Mnist


def get_dataloader(dataset,batch_size,shuffle=True):
    '''输入数据，获取一个数据加载器（分批加载）'''
    dataLoader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    return dataLoader
