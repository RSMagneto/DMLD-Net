import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
from thop import profile
from torch_sobel import sobel_conv


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    raise 'CUDA is not available'


class FEB(nn.Module):
    def __init__(self, input_size, output_size=32, ker_size=3, stride=1):
        super(FEB, self).__init__()
        self.conv_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())

        self.dilated_conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.LeakyReLU())

        self.dilated_conv2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.LeakyReLU())

        self.dilated_conv3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=3),
            nn.LeakyReLU())

        self.dilated_conv4 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=4),
            nn.LeakyReLU())

        self.conv_tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())
    def forward(self, data):
        conv_head = self.conv_head(data)
        x1, x2, x3, x4 = torch.split(conv_head, 8, dim=1)
        x1 = self.dilated_conv1(x1)
        x2 = self.dilated_conv2(x2)
        x3 = self.dilated_conv3(x3)
        x4 = self.dilated_conv4(x4)
        cat = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv_tail(cat)
        output = conv_head + out
        return output

class LDB(nn.Module):
    def __init__(self):
        super(LDB, self).__init__()
    def forward(self, fea_panlow, fea_ms):
        fea_panlow, fea_ms = (fea_panlow/torch.max(fea_panlow)), (fea_ms/torch.max(fea_ms))
        edge_pan = sobel_conv(fea_panlow, 32)
        edge_ms = sobel_conv(fea_ms, 32)
        dis_map = (1-edge_pan)*(1-edge_ms) + edge_pan*edge_ms + 1 - torch.abs(edge_pan - edge_ms)
        return dis_map, edge_pan, edge_ms

class conv_block1(nn.Module):
    def __init__(self, input_size, output_size=32, ker_size=3, stride=1):
        super(conv_block1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())
    def forward(self, data):
        conv1 = self.conv1(data)
        bicubic = F.interpolate(conv1, scale_factor=2, mode='bicubic', align_corners=True)
        out = self.conv2(bicubic)
        return out


class conv_block2(nn.Module):
    def __init__(self, input_size, output_size=64, ker_size=3, stride=1):
        super(conv_block2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())

    def forward(self, data):
        conv1 = self.conv1(data)
        bicubic = F.interpolate(conv1, size=(192, 192), mode='bicubic', align_corners=True)
        out = self.conv2(bicubic)
        return out


class conv_block3(nn.Module):
    def __init__(self, input_size, output_size=64, ker_size=3, stride=1):
        super(conv_block3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())

    def forward(self, data):
        conv1 = self.conv1(data)
        bicubic = F.interpolate(conv1, size=(256, 256), mode='bicubic', align_corners=True)
        out = self.conv2(bicubic)
        return out


class reconstruction(nn.Module):
    def __init__(self, input_size, output_size, ker_size=3, stride=1):
        super(reconstruction, self).__init__()
        self.conv = nn.Sequential(
        nn.ReflectionPad2d(1),
        torch.nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=ker_size, stride=stride, padding=0),
        torch.nn.LeakyReLU(),
        nn.ReflectionPad2d(1),
        torch.nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=ker_size, stride=stride, padding=0),
        torch.nn.LeakyReLU(),
        nn.ReflectionPad2d(1),
        torch.nn.Conv2d(in_channels=32, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
        torch.nn.LeakyReLU())
    def forward(self, data):
        out = self.conv(data)
        return out


class conv_solo(nn.Module):
    def __init__(self, input_size, output_size, ker_size=3, stride=1):
        super(conv_solo, self).__init__()
        self.conv = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0))
    def forward(self, data):
        out = self.conv(data)
        return out


class EGDU(nn.Module):
    def __init__(self):
        super(EGDU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        # self.batchnorm = nn.BatchNorm2d()
    def forward(self, fea_pan, dis_map):
        fea_pan = fea_pan/torch.max(fea_pan)
        edge_pan = sobel_conv(fea_pan, 32)
        conv3 = self.conv3(edge_pan)
        dis_map = self.conv1(dis_map) 
        bicubic = F.interpolate(dis_map, scale_factor=2, mode='bicubic', align_corners=True) # 2 32 128 128
        conv2 = self.conv2(bicubic) 
        conv4 = self.conv4(conv3 * bicubic) 
        out = conv2 + conv3 + conv4
        return out

class EGDU2(nn.Module):
    def __init__(self):
        super(EGDU2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        # self.batchnorm = nn.BatchNorm2d()
    def forward(self, fea_pan, dis_map):
        fea_pan = fea_pan/torch.max(fea_pan)
        edge_pan = sobel_conv(fea_pan, 32)
        conv3 = self.conv3(edge_pan)
        dis_map = self.conv1(dis_map) 
        bicubic = F.interpolate(dis_map, size=(192, 192), mode='bicubic', align_corners=True) 
        conv2 = self.conv2(bicubic) 
        conv4 = self.conv4(conv3 * bicubic) 
        out = conv2 + conv3 + conv4
        return out

class EGDU3(nn.Module):
    def __init__(self):
        super(EGDU3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        # self.batchnorm = nn.BatchNorm2d()
    def forward(self, fea_pan, dis_map):
        fea_pan = fea_pan/torch.max(fea_pan)
        edge_pan = sobel_conv(fea_pan, 32)
        conv3 = self.conv3(edge_pan)
        dis_map = self.conv1(dis_map) 
        bicubic = F.interpolate(dis_map, size=(256, 256), mode='bicubic', align_corners=True) 
        conv2 = self.conv2(bicubic) 
        conv4 = self.conv4(conv3 * bicubic) 
        out = conv2 + conv3 + conv4
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        )

        self.relu = nn.LeakyReLU()
    def forward(self, data):
        out = self.conv(data)
        out = self.relu(out + data)
        return out


class LocalDissimilarity(nn.Module):
    def __init__(self):
        super(LocalDissimilarity, self).__init__()
        self.FEB_pan = FEB(input_size=1)
        self.FEB_ms = FEB(input_size=4)
        self.FEB_mid1 = FEB(input_size=32)
        self.FEB_mid2 = FEB(input_size=32)

        self.LDB = LDB()
        self.EGDU = EGDU()
        self.EGDU2 = EGDU2()
        self.EGDU3 = EGDU3()

        self.conv_block1 = conv_block1(input_size=32)
        self.conv_block2 = conv_block2(input_size=32)
        self.conv_block3 = conv_block3(input_size=64)

        self.reconstruction = reconstruction(input_size=64, output_size=4) 
        # self.pooling = torch.nn.AdaptiveAvgPool2d(2)  # 128 or 32  # 128 or 32
        self.pooling1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((192, 192))
            )  # 128 or 32
        self.pooling2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((128, 128))
        )
        self.conv_solo1 = conv_solo(input_size=32, output_size=32)
        self.conv_solo2 = conv_solo(input_size=32, output_size=64)
        self.conv_solo3 = conv_solo(input_size=32, output_size=64)


    def forward(self, pan, ms):
        pan_low = F.interpolate(pan, size=(ms.shape[2], ms.shape[3]), mode='bicubic', align_corners=True)
        fea_pan = self.FEB_pan(pan)  
        fea_panlow = self.FEB_pan(pan_low)  
        fea_ms = self.FEB_ms(ms) 
        conv_block1 = self.conv_block1(fea_ms)
        dissimilaritymap1, edge_pan, edge_ms = self.LDB(fea_panlow, fea_ms) 
        pool1 = self.pooling1(fea_pan) 
        fea_pan1 = self.FEB_mid1(pool1) 

        pool2 = self.pooling2(fea_pan1)  
        fea_pan2 = self.FEB_mid2(pool2)  

        EGDU_out1 = self.EGDU(fea_pan2, dissimilaritymap1) 
        Conv1 = self.conv_solo1(EGDU_out1 * fea_pan2) 
        conv_block2 = self.conv_block2(conv_block1 + Conv1) 
        dissimilaritymap2, edge_pan2, edge_ms2 = self.LDB(fea_pan2, EGDU_out1) 
        EGDU_out2 = self.EGDU2(fea_pan1, dissimilaritymap2) 
        Conv2 = self.conv_solo2(EGDU_out2 * fea_pan1) 
        conv_block3 = self.conv_block3(conv_block2 + Conv2)  
        dissimilaritymap3, edge_pan3, edge_ms3 = self.LDB(fea_pan1, EGDU_out2)  
        EGDU_out3 = self.EGDU3(fea_pan, dissimilaritymap3) 
        Conv3 = self.conv_solo3(EGDU_out3 * fea_pan) 
        result = self.reconstruction(Conv3+conv_block3)
        return result, dissimilaritymap1, edge_pan, edge_ms


if __name__ == '__main__':
    model = LocalDissimilarity().cuda()
    ms = torch.randn(2, 4, 64, 64).cuda()
    pan = torch.randn(2, 1, 256, 256).cuda()
    result, dissimilaritymap, edge_pan, edge_ms = model(pan, ms)
    flops, params = profile(model, (pan, ms))
    print('flops:' + str(flops / 1000 ** 3) + 'G', 'parameters:' + str(params / 1000 ** 2) + 'M')
    print('ok')
    # flops, params = profile(model, (pan, ms))
    # print('flops:' + str(flops / 1000 ** 3) + 'G', 'parameters:' + str(params / 1000 ** 2) + 'M')