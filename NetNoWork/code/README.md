# LiuXiao 刘潇

自建包(还没有完善)：

  Adversary：封装对抗攻击方法
  
  Data_Reader：封装数据读取方法
  
  Models：封装常用模型
  
  Optimizer：封装优化方法
  
  Tools:封装一些工具
  
  _Base_ :其他包调用的公用方法

---------------------------------------------------------------

train1_clear.py 干净样本训练方法

train1_clear.pkl 训练好的模型

train2_InputZero.py 一个想法：训练时最小化输入的梯度(有点用但没有对抗训练效果好)

train2_InputZero.pkl 训练好的模型

train3_AdvT.pkl 对抗训练

train3_AdvT.py 训练好的模型

train4_AdvT_InputZero.py   train2+train3的结合

adversarial_attack.py 进行对抗样本攻击
