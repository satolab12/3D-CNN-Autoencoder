import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import skvideo.io
import glob

from lib import ParseGRU,Visualizer
from network import ThreeD_conv

parse = ParseGRU()
opt = parse.args
autoencoder = ThreeD_conv(opt)
autoencoder.train()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(),
                             lr=opt.learning_rate,
                             weight_decay=1e-5)

files = glob.glob(opt.dataset+'/*')
videos = [ skvideo.io.vread(file) for file in files ]
videos =  [video.transpose(3, 0, 1, 2) / 255.0 for video in videos ]
n_videos = len(videos)

def transform(video):
    trans_video = torch.empty(opt.n_channels,opt.T,opt.image_size,opt.image_size)
    for i in range(opt.T):
        img = video[:,i]
        img = trans(img).reshape(opt.n_channels,opt.image_size,opt.image_size)
        trans_video[:,i] = img
    return trans_video

def trim(video):
    start = np.random.randint(0, video.shape[1] - (opt.T+1))
    end = start + opt.T
    return video[:, start:end, :, :]

def random_choice():
    X = []
    for _ in range(opt.batch_size):
        video = videos[np.random.randint(0, n_videos-1)]
        video = torch.Tensor(trim(video))#video has (C,T,img,img)

        video = transform(video)
        X.append(video)
    X = torch.stack(X)
    return X

if opt.cuda:
    autoencoder.cuda()

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(1),
    transforms.Resize((opt.image_size,opt.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ])

losses = np.zeros(opt.n_itrs)
visual = Visualizer(opt)

for itr in range(opt.n_itrs):

    real_videos = random_choice()
    x = real_videos

    if opt.cuda:
        x = Variable(x).cuda()
    else:
        x = Variable(x)

    xhat = autoencoder(x)

    loss = mse_loss(xhat, x)
    losses[itr] = losses[itr] * (itr / (itr + 1.)) + loss.data * (1. / (itr + 1.))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('itr [{}/{}], loss: {:.4f}'.format(
        itr + 1,
        opt.n_itrs,
        loss))
    visual.losses = losses
    visual.plot_loss()

    if itr % opt.check_point == 0:
        tests = x[:opt.n_test].reshape(-1,opt.T,opt.n_channels,opt.image_size,opt.image_size)
        recon = xhat[:opt.n_test].reshape(-1,opt.T,opt.n_channels,opt.image_size,opt.image_size)

        for i in range(opt.n_test):
            #if itr == 0:
            save_image((tests[i]/2+0.5), os.path.join(opt.log_folder + '/generated_videos/3dconv', "real_itr{}_no{}.png" .format(itr,i)))
            save_image((recon[i]/2+0.5), os.path.join(opt.log_folder+'/generated_videos/3dconv', "recon_itr{}_no{}.png" .format(itr,i)))
            #torch.save(autoencoder.state_dict(), os.path.join('./weights', 'G_itr{:04d}.pth'.format(itr+1)))


