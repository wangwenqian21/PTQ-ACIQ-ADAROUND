## 文件组成

- ckpt：每个模型adaround训练过程存储的ckpt文件
- logs：每个模型的log文件，记录了每一层训练的alpha初值、终值，reconloss的初值、终值等信息。
- quantization：实现QConv2d,QConv2dBn等量化层类来替换原始conv2d和linear层
- util：主readme有介绍
- utils.py：全部的工具函数，包括adaround训练过程的实现（参考了aimet），插入量化层等
- act.py：以字典的形式存放所有后面跟了激活函数的（层名，激活函数）。参考了aciq的mark_relu实现