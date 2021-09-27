import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataloader
from my_classes import Dataset
from Loss import Loss

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--battery', type=int, default=15, metavar='N', help='Battery range in miles')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
battery = args.battery

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if args.cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sigmoid_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.sigmoid(y / temperature, dim=-1)


def gumbel_sigmoid(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_sigmoid_sample(logits, temperature)
    return y



class graph_augmentation(nn.Module):
    def __init__(self, temp, W):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(self.num_states, self.num_states)
        self.fc1.bias.data.fill_(self.edit_bias_init_mu)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.W = W


    def forward(self, x, temp):
        null_input = self.fc1(torch.zeros_like(x[0]))
        R = gumbel_sigmoid(null_input)
        # mask = np.ones((1,self.num_states))
        # mask[0,self.num_states-1]=0

        vgs = []

        for g in range(self.num_groups):
            y = x[g]
            vg = []
            for i in range(T):
                #r = kl.Dot(axes=-1)([s,R])
                r = torch.sum(y*R,axis=-1,keepdim=True)
                v = self.gamma**i*r
                vg.append(v)

                
                s = s-R*s
                
                logit = torch.mm(self.W, s)
                s = torch.abs(logit)/torch.sum(logit, axis=-1, keepdim=True)

            vgs.append(torch.sum(vg))

        vs = torch.cat(vgs)

        return vs, R




def train(epoch, loss_class, loader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(loader):
        if args.cuda:
            data[0] = data[0].cuda()
            data[1] = data[1].cuda()
            data[2] = data[2].cuda()
        optimizer.zero_grad()
        vs, R = model(data, temp)
        loss = loss_class.loss_function(vs, R)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))
        
        loss_class.on_batch_end(epoch)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))



def run():
    loss_class = Loss(model = model)
    for epoch in range(1, args.epochs + 1):
        iterable_loc = Dataset()
        loader = DataLoader(iterable_loc, batch_size=args.batch_size, shuffle=True, 'num_workers': 2)
        train(epoch, loss_class, loader)
        loss_class.on_epoch_end(epoch)

W = torch.from_numpy('below_{}.npy'.format(battery))
model = graph_augmentation(args.temp, W)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
temp = args.temp

if __name__ == '__main__':
    run()
