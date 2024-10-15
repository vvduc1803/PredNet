import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader import MapData
from model.model import PredNet
from evaluation.evaluate import eval

num_epochs = 150
batch_size = 8
num_workers = 2

lr = 0.001 # if epoch < 75 else 0.0001
nt = 6 # num of time steps
nt_predict = 15 # num of predict time steps

E_channels = (6, 96, 192, 384)
R_channels = (3, 48, 96, 192)
R_up_channels = (48, 96, 192)

stack_sizes = (3, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filter_sizes = (3, 3, 3)
Ahat_filter_sizes = (3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3)

model = PredNet(R_channels, E_channels, R_up_channels, stack_sizes, R_stack_sizes, A_filter_sizes, Ahat_filter_sizes, R_filter_sizes, output_mode='error', start_eval=nt_predict).cuda()

layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]))
time_loss_weights = 1./(nt-2) * torch.ones(nt-1, 1)
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights)

DATA_DIR = '/home/ana/Study/Occupancy_flow/PredNet_Build/data'
root = '/home/ana/Study/Occupancy_flow/PredNet_Build'
load_checkpoint = True
model_path = '/home/ana/Study/Occupancy_flow/PredNet_Build/pretrained/95.pkl'

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_train.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_train.hkl')

kitti_train = MapData(train_file, train_sources, nt)
kitti_val = MapData(val_file, val_sources, nt, nt_predict=nt_predict)

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if load_checkpoint:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

def lr_scheduler(optimizer, epoch):
    if epoch < num_epochs //2:
        return optimizer
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        return optimizer

def saveCheckpoint(state_dict, fileName = None):
    '''save the checkpoint for both restarting and evaluating.'''
    torch.save(state_dict, fileName)

def val(epoch, model, val_loader):
    with torch.inference_mode():
        model.eval()
        is_l, mse_l, msed_l = [], [], []

        for i, (inputs, test) in enumerate(val_loader):
            inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
            test = test.permute(0, 1, 4, 2, 3)
            inputs = Variable(inputs.cuda())

            X_hat = model.predict(inputs)

            test = test.permute(0, 1, 3, 4, 2)
            X_hat = X_hat.permute(1, 0, 2, 3, 4)

            test = test.cpu().numpy().astype(np.float32)
            X_hat = X_hat.cpu().numpy().astype(np.float32)

            is_, _, mse, mesd = eval(test, X_hat)
            is_l.append(is_)
            mse_l.append(mse)
            msed_l.append(mesd)

        is_ = sum(is_l) / len(is_l)
        mse = sum(mse_l) / len(mse_l)
        mesd = sum(msed_l) / len(msed_l)

        print((f'Epoch eval: {epoch - 1}/{num_epochs}, IS: {is_}, MSE: {mse}, MESD: {mesd}'))

    model.train()

def train(epoch, model, train_loader):
    for i, inputs in enumerate(train_loader):
        inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
        inputs = Variable(inputs.cuda())

        errors = model(inputs)  # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, nt - 1).cpu(), time_loss_weights)  # batch*n_layers x 1

        errors = torch.mm(errors.view(loc_batch, -1).cpu(), layer_loss_weights)
        errors = torch.mean(errors)
        errors.backward()

        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f'Epoch: {epoch}/{num_epochs}, step: {i}/{len(kitti_train) // batch_size}, errors: {errors}')

if __name__=='__main__':
    ran = 47
    torch.manual_seed(ran)
    import random
    random.seed(ran)
    import numpy as np
    np.random.seed(ran)


    for epoch in range(start_epoch, num_epochs):
        optimizer = lr_scheduler(optimizer, epoch)

        train(epoch, model, train_loader)

        if (epoch) % 5== 0:
            val(epoch, model, val_loader)

        if epoch % 5 == 0:
            state_dict = {
                'epoch': (epoch),
                'tr_loss': 0,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            saveCheckpoint(state_dict, f'{root}/{epoch}.pkl')








