import torch
from torch import nn
from torch.autograd import Variable

class ThreeD_conv(nn.Module):
    def __init__(self, opt, ndf=64, ngpu=1):
        super(ThreeD_conv, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.z_dim = opt.z_dim
        self.T = opt.T
        self.image_size = opt.image_size
        self.n_channels = opt.n_channels
        self.conv_size = int(opt.image_size/16)

        self.encoder = nn.Sequential(
            nn.Conv3d(opt.n_channels, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int((ndf*8)*(self.T/16)*self.conv_size*self.conv_size),self.z_dim ),#6*6
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.z_dim,int((ndf*8)*(self.T/16)*self.conv_size*self.conv_size)),#6*6
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d((ndf*8), ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf*4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 2, ndf , 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf , opt.n_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        bs = input.size(0)

        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            feature = self.encoder(input)
            z = self.fc1(feature.view(bs,-1))
            feature = self.fc2(z).reshape(bs,self.ndf*8,int(self.T/16),self.conv_size,self.conv_size)
            output = self.decoder(feature).view(bs,self.n_channels,self.T,self.image_size,self.image_size)

        return output

