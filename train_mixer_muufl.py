import torch
import os
import cv2
import numpy as np
from skimage import io
import scipy.io as sio
from scipy.io import savemat
import torch.utils.data as dataf
from mixer_muufl import MLPMixer
# from ori_net import  MLPMixer
import time

### parm and dataloader
hsi_datapath1 = '/home/sd1/liuyuzhe/MUUFL/Data.mat' #hsi data
hsi_datapath = '/home/sd1/liuyuzhe/MUUFL/muufl_50.mat' # pca_data
train_path = '/home/sd1/liuyuzhe/MUUFL/mask_train_150.mat'
test_path = '/home/sd1/liuyuzhe/MUUFL/mask_test_150.mat'
lidar_datapath = '/home/sd1/liuyuzhe/MUUFL/lidar.mat'
num_class = 11
patchsize = 15
patchsize_lidar = 15
batchsize = 24
EPOCH = 500
# channel = 64
channel = 94
pad_width = np.floor(patchsize / 2)
pad_width = np.int_(pad_width)

train_label = sio.loadmat(train_path)
test_label = sio.loadmat(test_path)

hsi_data = sio.loadmat(hsi_datapath1)
hsi_data_pca = sio.loadmat(hsi_datapath)
lidar_data = sio.loadmat(lidar_datapath)

train_label = train_label['mask_train']
test_label = test_label['mask_test']
hsi_data = hsi_data['Hdata']
hsi_data_pca = hsi_data_pca['pca_data']
lidar_data = lidar_data['lidar']

lidar_data = lidar_data[:,:,0]
# train_label = io.imread(train_path)
# test_label = io.imread(test_path)
# hsi_data = io.imread(hsi_datapath)
lidar_data = lidar_data.astype(np.float32)

# hsi_data = np.concatenate((hsi_data,hsi_data_pca),axis=2)
###   data normlization
[m, n, l] = hsi_data.shape
for i in range(l):
    minimal = hsi_data[:, :, i].min()
    maximal = hsi_data[:, :, i].max()
    hsi_data[:, :, i] = (hsi_data[:, :, i] - minimal) / (maximal - minimal)

hsi_data = np.concatenate((hsi_data,hsi_data_pca),axis=2)
x = np.empty((339, 234, channel), dtype='float32')
for i in range(channel):
    temp = hsi_data[:, :, i]
    pad_width = np.floor(patchsize / 2)
    pad_width = np.int_(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x[:, :, i] = temp2
# x = np.concatenate((x, ))
[h,w] =  lidar_data.shape
min = lidar_data.min()
max = lidar_data.max()
lidar_data = (lidar_data-min)/ (max - min)
temp1 = lidar_data
pad_lidar_width = np.floor(patchsize_lidar / 2)
pad_lidar_width = np.int_(pad_lidar_width)
temp_lidar = np.pad(temp1, pad_lidar_width, 'symmetric')
x_lidar = temp_lidar

print('done norm')

### data -> patch
[ind1, ind2] = np.where(train_label != 0)
TrainNum = len(ind1)
TrainPatch = np.empty((TrainNum, channel, patchsize, patchsize), dtype='float32')
Train_lidar_Patch = np.empty((TrainNum, 1, patchsize_lidar, patchsize_lidar), dtype='float32')
TrainLabel = np.empty(TrainNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
ind5 = ind1 + pad_lidar_width
ind6 = ind2 + pad_lidar_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = np.reshape(patch, (patchsize * patchsize, channel))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (channel, patchsize, patchsize))
    TrainPatch[i, :, :, :] = patch
    patchlabel = train_label[ind1[i], ind2[i]]
    TrainLabel[i] = patchlabel

for i in range(len(ind1)):
    lidar_patch = x_lidar[(ind5[i] - pad_lidar_width):(ind5[i] + pad_lidar_width + 1), (ind6[i] - pad_lidar_width):(ind6[i] + pad_lidar_width + 1)]
    lidar_patch = np.reshape(lidar_patch, (patchsize_lidar * patchsize_lidar, 1))
    lidar_patch = np.transpose(lidar_patch)
    lidar_patch = np.reshape(lidar_patch, (1, patchsize_lidar, patchsize_lidar))
    Train_lidar_Patch[i, :, :, :] = lidar_patch
    # patchlabel = train_label[ind1[i], ind2[i]]
    # TrainLabel[i] = patchlabel
[ind1, ind2] = np.where(test_label != 0)
np.save('index.npy', [ind1, ind2])
TestNum = len(ind1)
Test_lidar_Patch = np.empty((TestNum, 1, patchsize_lidar, patchsize_lidar), dtype='float32')
TestPatch = np.empty((TestNum, channel, patchsize, patchsize), dtype='float32')
TestLabel = np.empty(TestNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
ind5 = ind1 + pad_lidar_width
ind6 = ind2 + pad_lidar_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = np.reshape(patch, (patchsize * patchsize, channel))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (channel, patchsize, patchsize))
    TestPatch[i, :, :, :] = patch
    testpatchlabel = test_label[ind1[i], ind2[i]]
    TestLabel[i] = testpatchlabel

for i in range(len(ind1)):
    lidar_patch = x_lidar[(ind5[i] - pad_lidar_width):(ind5[i] + pad_lidar_width + 1), (ind6[i] - pad_lidar_width):(ind6[i] + pad_lidar_width + 1)]
    lidar_patch = np.reshape(lidar_patch, (patchsize_lidar * patchsize_lidar, 1))
    lidar_patch = np.transpose(lidar_patch)
    lidar_patch = np.reshape(lidar_patch, (1, patchsize_lidar, patchsize_lidar))
    Test_lidar_Patch[i, :, :, :] = lidar_patch
    # testpatchlabel = test_label[ind1[i], ind2[i]]
    # TestLabel[i] = testpatchlabel
Train_lidar_Patch = torch.from_numpy(Train_lidar_Patch)
TrainPatch = torch.from_numpy(TrainPatch)
TrainLabel = torch.from_numpy(TrainLabel) - 1
TrainLabel = TrainLabel.long()

Test_lidar_Patch = torch.from_numpy(Test_lidar_Patch)
TestPatch = torch.from_numpy(TestPatch)
TestLabel = torch.from_numpy(TestLabel) - 1
TestLabel = TestLabel.long()
print('done patch')
dataset = dataf.TensorDataset(TrainPatch,Train_lidar_Patch, TrainLabel)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)


# APExp = AttributionPriorExplainer(dataset, batchsize,k=1)


### model_build
model = MLPMixer(in_channels=94, image_size=15, patch_size=1, num_classes=11,
                     dim=128, depth=2, token_dim=256, channel_dim=256,lidar_size=15)
# model = MLPMixer(in_channels=63, image_size=11, patch_size=1, num_classes=6,
#                      dim=128, depth=4, token_dim=128, channel_dim=512)
model = torch.nn.DataParallel(model)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=50, last_epoch=-1)
loss_func = torch.nn.CrossEntropyLoss()
torch.cuda.synchronize()
### train_model
bestacc = 0
start = time.time()
for epoch in range(EPOCH):
    for step, (x1,x2,y) in enumerate(train_loader):
        # move train data to GPU
        x1 = x1.cuda()
        x2 = x2.cuda()
        y = y.cuda()
        output = model(x1,x2)[0]
        loss = loss_func(output,y)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        # scheduler.step()
        optimizer.step()  # apply gradients
        Classes = np.unique(train_label)
        if step % 50 == 0:
            model.eval()
            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 5000
            for i in range(number):
                temp = TestPatch[i * 5000:(i + 1) * 5000, :, :, :]
                temp_lidar = Test_lidar_Patch[i * 5000:(i + 1) * 5000, :, :, :]
                temp_lidar = temp_lidar.cuda()
                temp = temp.cuda()
                # temp3 = w2*cnn(temp1, temp, temp2)[1] + w0*cnn(temp1, temp, temp2)[0]
                temp3 = model(temp, temp_lidar)[0]
                temp3 = torch.max(temp3, 1)[1].squeeze()
                pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
                del temp, temp_lidar,temp3#shengxia testpatch 2 de shujv

            if (i + 1) * 5000 < len(TestLabel):
                temp = TestPatch[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp_lidar = Test_lidar_Patch[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp = temp.cuda()
                temp_lidar = temp_lidar.cuda()
                # temp3 = w2*cnn(temp1, temp, temp2)[1] + w0*cnn(temp1, temp, temp2)[0]
                temp3 = model(temp, temp_lidar)[0]
                temp3 = torch.max(temp3, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(TestLabel)] = temp3.cpu()

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
            diedai =  loss.data.cpu().numpy()
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % accuracy)

            # save the parameters in network
            if accuracy > bestacc:
                torch.save(model.state_dict(), 'muufl_weight.pth')
                bestacc = accuracy
            model.train()

torch.cuda.synchronize()
end = time.time()
print(end - start)
Train_time = end - start
###  test_model

model.load_state_dict(torch.load('muufl_weight.pth'))
model.eval()
torch.cuda.synchronize()
start = time.time()
pred_y = np.empty((len(TestLabel)), dtype='float32')
pre_tsne = np.empty(shape=[5000,64])
number = len(TestLabel)//5000
for i in range(number):
    temp = TestPatch[i*5000:(i+1)*5000, :, :]
    temp_lidar = Test_lidar_Patch[i*5000:(i+1)*5000, :, :]
    temp_lidar = temp_lidar.cuda()
    temp = temp.cuda()
    # temp3 = w2 * cnn(temp1, temp, temp2)[1] + w0 * cnn(temp1, temp, temp2)[0]
    temp3,pre_fea = model(temp, temp_lidar)
    temp3 = torch.max(temp3, 1)[1].squeeze()
    pre_tsne = np.append(pre_tsne, pre_fea.cpu().detach().numpy(), axis=0)
    pred_y[i*5000:(i+1)*5000] = temp3.cpu()
    del temp,temp_lidar, temp3,pre_fea

if (i+1)*5000 < len(TestLabel):
    temp = TestPatch[(i+1)*5000:len(TestLabel), :, :]
    temp_lidar = Test_lidar_Patch[(i+1)*5000:len(TestLabel), :, :]
    temp_lidar = temp_lidar.cuda()
    temp = temp.cuda()
    # temp3 = w2 * cnn(temp1, temp, temp2)[1] + w0 * cnn(temp1, temp, temp2)[0]
    temp3,pre_fea = model(temp, temp_lidar)
    temp3 = torch.max(temp3, 1)[1].squeeze()
    pred_y[(i+1)*5000:len(TestLabel)] = temp3.cpu()
    pre_tsne = np.append(pre_tsne, pre_fea.cpu().detach().numpy(), axis=0)
    del temp, temp3,pre_fea


pre_tsne = pre_tsne[5000:,:]
savemat('matrix_mul_mlp.mat',{'tsne':pre_tsne})
savemat('matrix_mul_mlp_label.mat',{'label':pred_y})

pred_y = torch.from_numpy(pred_y).long()


pred_q = pred_y + 1
index = np.load('index.npy')
pred_map = np.zeros_like(test_label)
for i in range(index.shape[1]):
    pred_map[index[0, i], index[1, i]] = pred_q[i]

pred_final = pred_map
cv2.imwrite('pred_mul.png',pred_final)
np.save('pred_mul.npy', pred_final)


def confusion(pred, label):
    mx = np.zeros((num_class,num_class))
    for i in range (TestNum):
        mx[pred[i], label[i]] += 1
    mx = np.asarray(mx, dtype=np.int16)
    np.savetxt("confusion.txt", mx, delimiter="", fmt="%s")
    return mx


def oa_kappa(confusion):
    N = np.sum(confusion)
    N_ober = np.trace(confusion)
    Po = 1.0*N_ober / N
    h_sum = np.sum(confusion,axis=0)
    v_sum = np.sum(confusion,axis=1)
    Pe = np.sum(np.multiply(1.0*h_sum/N, 1.0*v_sum/N))
    kappa = (Po - Pe)/(1.0 - Pe)
    return kappa

con_matrix = confusion(pred_y, TestLabel)
Ka = oa_kappa(con_matrix)
OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)

Classes = np.unique(TestLabel)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestLabel)):
        if TestLabel[j] == cla:
            sum += 1
        if TestLabel[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()

AA = np.mean(EachAcc)
print('ka is', Ka)
print('AA is', AA)
print(OA)
print(EachAcc)

torch.cuda.synchronize()
end = time.time()
print(end - start)
Test_time = end - start
Final_OA = OA

print('The OA is: ', Final_OA)
print('The Training time is: ', Train_time)
print('The Test time is: ', Test_time)