
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from my_classes import Dataset
from Loss import Loss

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--battery', type=int, default=15, metavar='N', help='Battery range in miles')
parser.add_argument('--time', type=int, default=4, metavar='N', help='maximum diffusion step')
parser.add_argument('--gamma', type=float, default=0.99, metavar='S', help='discount factor')

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
    return F.sigmoid(y / temperature)


def gumbel_sigmoid(logits, temperature=args.temp):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_sigmoid_sample(logits, temperature)
    return y



class graph_augmentation(nn.Module):
    def __init__(self, temp, W, edit_bias):
        super(graph_augmentation, self).__init__()
        self.edit_bias_init_mu = edit_bias
        self.num_states = W.shape[0]
        self.fc1 = nn.Linear(self.num_states, self.num_states)
        self.fc1.bias.data.fill_(self.edit_bias_init_mu)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.W = W
        self.num_groups = 3


    def forward(self, x, temp):
        x1, x2, x3 = x
        null_input = self.fc1(torch.zeros_like(x1).float())
        R = gumbel_sigmoid(null_input)
        # mask = np.ones((1,self.num_states))
        # mask[0,self.num_states-1]=0

        vgs = []
        y = [x1, x2, x3]
        for g in range(self.num_groups):
            s = y[g]
            vg = 0
            for i in range(args.time):
                #r = kl.Dot(axes=-1)([s,R])
                r = torch.sum(s*R,axis=-1,keepdim=True)
                v = args.gamma**i*r
                vg += v

                
                s = s-R*s
                
                logit = torch.mm(self.W, torch.transpose(s, 0, 1))
                logit = torch.transpose(logit, 0, 1)
                s = torch.abs(logit)/torch.sum(logit, axis=-1, keepdim=True)

            vgs.append(vg)

        vs = torch.cat(vgs, dim=-1)	

        return vs, R




def train(epoch, loss_class, loader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(loader):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        vs, R = model(data, temp)
        loss = loss_class.loss_function(vs, R)
        loss.mean().backward()
        train_loss += loss.mean().item() * len(data)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                       100. * batch_idx / len(loader),
                       loss.mean().item()))
        
        outputs = [vs, R]
        loss_class.on_batch_end(epoch, outputs)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(loader.dataset)))



def run():
    loss_class = Loss()
    for epoch in range(1, args.epochs + 1):
        iterable_loc = Dataset()
        loader = DataLoader(iterable_loc, batch_size=args.batch_size, shuffle=True)
        train(epoch, loss_class, loader)
        loss_class.on_epoch_end(epoch)

W = torch.from_numpy(np.load('below_{}.npy'.format(battery)))
edit_mu = 0
model = graph_augmentation(args.temp, W, edit_mu)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
temp = args.temp

if __name__ == '__main__':
    run()
