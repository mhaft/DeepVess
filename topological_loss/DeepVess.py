# Copyright 2019-2020, Mohammad Haft-Javaherian. (mh973@cornell.edu).
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#   References:
#   -----------
#   [1] Haft-Javaherian, M., Villiger, M., Schaffer, C. B., Nishimura, N., Golland, P., & Bouma, B. E. (2020).
#       A Topological Encoding Convolutional Neural Network for Segmentation of 3D Multiphoton Images of Brain
#       Vasculature Using Persistent Homology. In Proceedings of the IEEE/CVF Conference on Computer Vision and
#       Pattern Recognition Workshops (pp. 990-991).
#       http://openaccess.thecvf.com/content_CVPRW_2020/html/w57/Haft-Javaherian_A_Topological_Encoding_Convolutional_Neural_Network_for_Segmentation_of_3D_CVPRW_2020_paper.html
# =============================================================================

from __future__ import print_function

from builtins import input
import sys
import time
from random import shuffle
import itertools as it
import glob

import numpy as np
from six.moves import range
import h5py
import scipy.io as io
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
from tqdm import tqdm

lr = 1e-4
# Change isTrain to True if you want to train the network
isTrain = True
# Change isForward to True if you want to test the network
isForward = True
# padSize is the padding around the central voxel to generate the field of view
padSize = ((3, 3), (48, 49), (48, 49), (0, 0))
WindowSize = np.sum(padSize, axis=1) + 1
# pad Size around the central voxel to generate 2D region of interest
corePadSize = 10
# number of epoch to train
nEpoch = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
# The input h5 file location and batch size
inputData = sys.argv[1] if len(sys.argv) > 1 else input("Enter h5 input file path (e.g. ../a.h5)> ")
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

# Import Data
f = h5py.File(inputData, 'r')
im = np.array(f.get('/im'))
im = im.reshape(im.shape + (1,))
imSize = im.size
imShape = im.shape
if isTrain:
    l = np.array(f.get('/l'))
    l = l.reshape(l.shape + (1,))
    nc = im.shape[1]
    tst = im[:, :(nc // 4), :]
    tstL = l[:, :(nc // 4), :]
    trn = im[:, (nc // 2):, :]
    trnL = l[:, (nc // 2):, :]
    tst = np.pad(tst, padSize, 'symmetric')
    trn = np.pad(trn, padSize, 'symmetric')
if isForward:
    im = np.pad(im, padSize, 'symmetric')
    V = np.zeros(imShape, dtype=np.float32)
print("Data loaded.")


class Dataset(data.Dataset):
    def __init__(self, im, l, imShape, WindowSize, corePadSize, isTrain=False, offset=None):
        self.im, self.l = im, l
        self.imShape, self.WindowSize, self.corePadSize = imShape, WindowSize, corePadSize
        self.sampleID, self.isTrain, self.offset = [], isTrain, offset if offset is not None else (0, 0)
        self.__shuffle__()

    def __shuffle__(self):
        self.sampleID = []
        if self.isTrain:
            self.offset = np.random.randint(0, 2 * self.corePadSize, 2)
        for i in range(0, imShape[0]):
            for j in it.chain(range(self.corePadSize+ self.offset[0], self.imShape[1] - self.corePadSize,
                                    2 * self.corePadSize + 1), [self.imShape[1] - self.corePadSize - 1]):
                for k in it.chain(range(self.corePadSize+ self.offset[1], self.imShape[2] - self.corePadSize,
                                        2 * self.corePadSize + 1), [self.imShape[2] - self.corePadSize - 1]):
                    self.sampleID.append(np.ravel_multi_index((i, j, k, 0), self.imShape))
        if self.isTrain:
            shuffle(self.sampleID)

    def __len__(self):
        return len(self.sampleID)

    def __getitem__(self, index):
        """Generates one sample of data"""
        ID = self.sampleID[index]
        r = np.unravel_index(ID, self.imShape)
        im_ = self.im[r[0]:(r[0] + self.WindowSize[0]), r[1]:(r[1] + self.WindowSize[1]),
                      r[2]:(r[2] + self.WindowSize[2]), :].transpose((-1, 0, 1, 2)).copy()
        if self.isTrain:
            im_ = np.clip(0.1 * (np.random.rand() - 0.5) +
                          im_ * (1 + 0.20 * (np.random.rand() - 0.5)), -.5, .5)
        if self.l is not None:
            l_ = self.l[r[0], (r[1] - self.corePadSize):(r[1] + self.corePadSize + 1),
                        (r[2] - self.corePadSize):(r[2] + self.corePadSize + 1), 0].flatten().astype('int64')
        return im_, l_ if self.l is not None else ID


class DeepVess(nn.Module):
    def __init__(self):
        super(DeepVess, self).__init__()

        self.activ = nn.LeakyReLU()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, 3),
            self.activ ,
            nn.Conv3d(32, 32, 3),
            self.activ ,
            nn.Conv3d(32, 32, 3),
            self.activ ,
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),
            self.activ,
            nn.Conv2d(64, 64, (3, 3)),
            self.activ,
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 21 * 21, 1024),
            self.activ ,
            nn.Dropout(),
            nn.Linear(1024, 2 * 1 * 21 * 21)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[:2] + x.shape[3:])
        x = self.conv2(x)
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        x = self.fc(x)
        x = x.reshape(x.shape[0], 2, 21* 21)
        return x


class TopLoss(nn.Module):
    def __init__(self, size):
        super(TopLoss, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size,  sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=0)  # penalize more than 0 hole
        self.topfn2 = SumBarcodeLengths(dim=0)  # penalize more than 1 max

    def forward(self, beta):
        eps = 1e-7
        beta = torch.clamp(F.softmax(beta, dim=1), eps, 1 - eps)
        beta = beta[0:beta.shape[0] // 10, ...]
        loss = 0
        for i in range(beta.shape[0]):
            dgminfo = self.pdfn(beta[i, 1, :])
            loss += self.topfn(dgminfo) + self.topfn2(dgminfo)
        return loss / beta.shape[0]


def dice_loss(y, l):
    """loss function based on mulitclass Dice index"""
    eps = 1e-7
    l = l.type(torch.cuda.FloatTensor)
    y = torch.clamp(F.softmax(y, dim=1), eps, 1 - eps)[:, 1, :]
    yl, yy, ll = y * l, y * y, l * l
    return 1 - (2 *yl.sum() + eps) / (ll.sum() + yy.sum() + eps)


model = DeepVess()
model = nn.DataParallel(model)
model = model.cuda()
CE = torch.nn.CrossEntropyLoss().cuda()
tloss = TopLoss((2 * corePadSize + 1,) * 2).cuda()
Loss = lambda y_, l_: dice_loss(y_, l_).cuda() + CE(y_, l_) + tloss(y_) / 100
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lastEpoch = max([0, ] + [int(a[11:-3]) for a in glob.glob('model-epoch*.pt')])

if isTrain and lastEpoch < nEpoch:
    if lastEpoch == 0:
        file_log = open("model.log", "w")
        file_log.write("Epoch, Step, training accuracy, test accuracy, Time (hr) \n")
        file_log.close()
    else:
        checkpoint = torch.load("model-epoch" + str(lastEpoch) + ".pt")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print("model-epoch" + str(lastEpoch) + ".pt is loaded.")
    train_data = data.DataLoader(Dataset(trn, trnL, trnL.shape, WindowSize, corePadSize, isTrain=True),
                                 batch_size=batch_size)
    train_data_ = data.DataLoader(Dataset(trn, trnL, trnL.shape, WindowSize, corePadSize, isTrain=False),
                                 batch_size=batch_size)
    test_data = data.DataLoader(Dataset(tst, tstL, tstL.shape, WindowSize, corePadSize, isTrain=False),
                                batch_size=batch_size)
    start = time.time()
    numBatch = np.ceil(train_data.dataset.__len__() / batch_size).astype('int') 
    loss_running = np.ones((numBatch))
    for epoch in range(lastEpoch, nEpoch):
        model.train()
        train_data = data.DataLoader(Dataset(trn, trnL, trnL.shape, WindowSize, corePadSize, isTrain=True),
                                 batch_size=batch_size)
        numBatch = np.ceil(train_data.dataset.__len__() / batch_size).astype('int')
        tq = tqdm(enumerate(train_data), 'Training', numBatch)
        loss_running = []
        for i, d in tq:
            x1, l1 = d[0].cuda(), d[1].cuda(0)
            optimizer.zero_grad()
            out = model(x1)
            loss = Loss(out, l1)
            loss.backward()
            optimizer.step()
            loss_running.append(loss.item())
            tq.set_description('Training (Epoch %d/%d, running loss= %f, loss=%f)' %
                               (epoch + 1, nEpoch, np.nanmean(loss_running), loss_running[i]))
        if epoch % 10 == 9:
            model.eval()
            loss = []

            for i, d in enumerate(train_data_):
                x1, l1 = d[0].cuda(), d[1].cuda(0)
                out = model(x1)
                loss.append(Loss(out, l1).item())
            train_accuracy = np.mean(loss)
            loss = []
            for i, d in enumerate(test_data):
                x1, l1 = d[0].cuda(), d[1].cuda(0)
                out = model(x1)
                loss.append(Loss(out, l1).item())
            test_accuracy = np.mean(loss)
            end = time.time()
            print("epoch %d, training accuracy %g, test accuracy %g. %f hour to finish." %
                  (epoch + 1, train_accuracy, test_accuracy, (nEpoch - epoch - 1) / (epoch + 1 - lastEpoch + 1) * (end - start) / 3600))
            file_log = open("model.log", "a")
            file_log.write("%d, %f, %f, %f \n" % (epoch + 1, train_accuracy,
                                                      test_accuracy, (end - start) / 3600))
            file_log.close()

        if epoch % 10 == 9:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       "model-epoch" + str(epoch + 1) + ".pt")


if isForward:
    lastEpoch = max([0] + [int(a[11:-3]) for a in glob.glob('model-epoch*.pt')])
    checkpoint = torch.load("model-epoch" + str(nEpoch) + ".pt")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    model.eval()
    print("model-epoch" + str(nEpoch) + ".pt restored.")
    I = [0, int( 2 * corePadSize / 3 + 1), int(4 * corePadSize / 3 + 1)]
    for offset in [(ii, ij) for ii in I for ij in I]:
        Forward_data = data.DataLoader(Dataset(im, None, imShape, WindowSize, corePadSize, isTrain=False,
                                               offset=offset), batch_size=batch_size)
        numBatch = Forward_data.dataset.__len__() // batch_size + 1
        for i, d in tqdm(enumerate(Forward_data), 'Forward', numBatch):
            x1, vID = d[0].cuda(), d[1]
            y1 = np.reshape(np.argmax(model(x1).detach().cpu().numpy(), 1),
                            (-1, (2 * corePadSize + 1), (2 * corePadSize + 1)))
            for j in range(len(vID)):
                r = np.unravel_index(vID[j], imShape)
                V[r[0], (r[1] - corePadSize):(r[1] + corePadSize + 1),
                    (r[2] - corePadSize):(r[2] + corePadSize + 1), 0] += y1[j, ...]
    V = (V > (np.max(V) / 2))
    fn = inputData[:-3] + "-epoch" + str(nEpoch) + '-V_fwd.mat'
    io.savemat(fn, {'V': np.transpose(np.reshape(V, imShape[0:3]), (2, 1, 0))})
    print(fn + '- is saved.')

