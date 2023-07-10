import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    raise 'CUDA is not available'

def sobel_conv(data, channel):
    conv_op_x = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)

    sobel_kernel_y = torch.tensor([[-1,-2,-1],
                                   [ 0, 0, 0],
                                   [ 1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    conv_op_x.weight.data = sobel_kernel_x
    conv_op_y.weight.data = sobel_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5*abs(edge_x) + 0.5*abs(edge_y)
    return result


def prewitt_conv(data, channel):
    conv_op_x = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)

    sobel_kernel_y = torch.tensor([[-1,-1,-1],
                                   [ 0, 0, 0],
                                   [ 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    conv_op_x.weight.data = sobel_kernel_x
    conv_op_y.weight.data = sobel_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5*abs(edge_x) + 0.5*abs(edge_y)
    return result


def laplacian_conv(data, channel):
    conv_op = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    laplacian_kernel = torch.tensor([[ 0,  1, 0],
                                     [ 1, -4, 1],
                                     [ 0,  1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    conv_op.weight.data = laplacian_kernel
    result = conv_op(data)
    return result



