import torch
import params
import math

'''

model.py

Define our GAN model

The cube_len is 32x32x32, and the maximum number of feature map is 256, 
so the results may be inconsistent with the paper

'''

class net_G(torch.nn.Module):
    def __init__(self, args):
        super(net_G, self).__init__()
        self.args = args
        self.cube_len = args.cube_len
        self.bias = params.bias
        self.z_dim = params.z_dim
        self.f_dim = args.cube_len
        self.n_layers = int(math.log2(self.f_dim)) - 1 # 2^6=64, so 5 for f_dim 64, 6 for f_dim 128
        self.max_feature = math.pow(2, self.n_layers - 2) # 8 for f_dim 64, 16 for f_dim 128
        self.layers = []
                
        for i in range(0, self.n_layers):
            if i==0:
                layer = self.conv_layer(self.z_dim, self.max_feature, kernel_size=4, stride=2, padding=(0,0,0), bias=self.bias)
            elif i==self.n_layer -1:
                layer = torch.nn.Sequential(
                    torch.nn.ConvTranspose3d(self.f_dim, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
                    torch.nn.Sigmoid()
                    # torch.nn.Tanh()
                )
            else:
                cur_pow = math.pow(2, self.n_layers - 1 - i) # for 128, self.n_layers=6, then get 2^4,2^3...
                layer = self.conv_layer(cur_pow, int(cur_pow/2), kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
            self.layers.append(layer)


        # self.layer1 = self.conv_layer(self.z_dim, self.f_dim*8, kernel_size=4, stride=2, padding=padd, bias=self.bias)
        # if self.cube_len == 128:
        #     self.layer1 = self.conv_layer(self.z_dim, self.f_dim*16, kernel_size=4, stride=2, padding=padd, bias=self.bias)
        #     self.addlayer = self.conv_layer(self.f_dim*16, self.f_dim*8, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        
        # self.layer2 = self.conv_layer(self.f_dim*8, self.f_dim*4, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        # self.layer3 = self.conv_layer(self.f_dim*4, self.f_dim*2, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        # self.layer4 = self.conv_layer(self.f_dim*2, self.f_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        
        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.ConvTranspose3d(self.f_dim, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
        #     torch.nn.Sigmoid()
        #     # torch.nn.Tanh()
        # )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.ReLU(True)
            # torch.nn.LeakyReLU(self.leak_value, True)
        )
        return layer

    def forward(self, x):
        out = x.view(-1, self.z_dim, 1, 1, 1)
        # print("G1",out.size())  # torch.Size([batch_size, 200, 1, 1, 1])
        out = self.layer1(out)
        # print("G2",out.size())  # torch.Size([batch_size, f_dim*16, 4, 4, 4])
        if self.cube_len == 128:
            out = self.addlayer(out)

        out = self.layer2(out)
        # print("G3",out.size())  # torch.Size([batch_size, f_dim*8, 8, 8, 8])
        out = self.layer3(out)
        # print("G4",out.size())  # torch.Size([batch_size, f_dim*4, 16, 16, 16])
        out = self.layer4(out)
        # print("G5",out.size())  # torch.Size([batch_size, f_dim*2, 32, 32, 32])
        out = self.layer5(out)
        # print("G6",out.size())  # torch.Size([batch_size, f_dim, 64, 64, 64])
        out = torch.squeeze(out)
        return out


class net_D(torch.nn.Module):
    def __init__(self, args):
        super(net_D, self).__init__()
        self.args = args
        self.cube_len = args.cube_len
        self.leak_value = params.leak_value
        self.bias = params.bias

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.f_dim = params.cube_len

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.f_dim*8, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.Sigmoid()
        )

        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Linear(256*2*2*2, 1),
        #     torch.nn.Sigmoid()
        # )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

    def forward(self, x):
        # out = torch.unsqueeze(x, dim=1)
        # print("Before Descriminator", x.size())
        out = x.view(-1, 1, self.f_dim, self.f_dim, self.f_dim)
        # print("D1",out.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        out = self.layer1(out)
        # print("D2",out.size())  # torch.Size([batch_size, 1, 32, 32, 32])
        out = self.layer2(out)
        # print("D3",out.size())  # torch.Size([batch_size, f_dim, 16, 16, 16])
        out = self.layer3(out)
        # print("D4",out.size())  # torch.Size([batch_size, f_dim*2, 8, 8, 8])
        out = self.layer4(out)
        # print("D5",out.size())  # torch.Size([batch_size, f_dim*8, 4, 4, 4])
        # out = out.view(-1, 256*2*2*2)
        # print("D6",out.size())  # torch.Size([batch_size, f_dim*8, 4, 4, 4])
        out = self.layer5(out)
        # print("D7",out.size())  # torch.Size([batch_size, 1, 1, 1, 1])
        out = torch.squeeze(out)
        return out

