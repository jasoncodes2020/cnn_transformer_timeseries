import torch
# from torchstat import stat
from attention import CNNLSTMModel_SE, CNNLSTMModel,CNNLSTMModel_HW,CNNLSTMModel_CBAM,CNNLSTMModel_ECA
# import torchvision.models as models
# # net = models.vgg11()
# net = CNNLSTMModel()
# stat(net,(16, 4, 1))    # (3,224,224)表示输入图片的尺寸
from torchvision.models import resnet50
# from thop import profile
#
# model = CNNLSTMModel()
# input = torch.randn(16, 1, 4)
# flops, params = profile(model, inputs=(input, ))
# -- coding: utf-8 --
import torchvision
# from ptflops import get_model_complexity_info
#
# # model = torchvision.models.alexnet(pretrained=False)
# model = CNNLSTMModel()
# flops, params = get_model_complexity_info(model, (16, 1, 4))
# print('flops: ', flops, 'params: ', params)
# -- coding: utf-8 --
import torch
import torchvision
from thop import profile

# Model
print('==> Building model..')
# model = torchvision.models.alexnet(pretrained=False)
model = CNNLSTMModel()
dummy_input = torch.randn(16, 1, 4)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
