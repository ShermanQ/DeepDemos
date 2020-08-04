import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models,transforms
from PIL import Image
from torch.autograd import Variable

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), 128, kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),

                                      nn.Conv2d(128, 32, kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),

                                      nn.Conv2d(32, int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),

                                      )

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)

        return outputs


class ScenceCNN(nn.Module):
    def __init__(self, img_path='/workspace/tian/scripts/plot/background.png'):
        super(ScenceCNN, self).__init__()
        self.img_path = img_path
        self.conv2batch = conv2DBatchNormRelu(in_channels=512, n_filters=1, k_size=1, stride=1, padding=0)

        self.model = models.vgg16(pretrained=True)

    def forward(self, seq):
        Vgg16 = self.model

        for p in Vgg16.parameters():
            p.requires_grad = False

        self.seq = int(seq)
        img = Image.open(self.img_path)
        # img=img.resize((224,224),Image.BICUBIC)
        transforms1 = transforms.Compose([transforms.Scale([224, 224]), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        img_tensor = transforms1(img)

        # img_tensor =Variable(img_tensor.cuda())

        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        print("sbbbbb")
        result = Vgg16.features.forward(img_tensor.cuda())
        print("sure sbbbbb")
        # print("result:",result.size())
        # result = result.squeeze(0)
        result = F.upsample(result, size=((self.seq * self.seq), 48), mode="bilinear", align_corners=True)

        conv = self.conv2batch(result).squeeze()

        result = Variable(conv)
        return result
def test():
    net = ScenceCNN()
    # img_path = '/workspace/tian/scripts/plot/background.png'
    # img = Image.open(img_path)
    # # img=img.resize((224,224),Image.BICUBIC)
    # transforms1 = transforms.Compose([transforms.Scale([224, 224]), transforms.ToTensor(),
    #                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # img_tensor = transforms1(img)
    y = net(200)
    print(y.size())
test()