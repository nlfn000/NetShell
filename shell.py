import os
from datetime import datetime

import torch
import torchvision
from torch import optim
from torch.autograd import Variable
from torchvision.transforms import transforms

from blueprints.LeNet import LeNet


class NetShell:
    default_scale = 224
    default_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def sav_loader(path=None, auto_load_dir=None):
        if auto_load_dir:
            files = os.listdir(auto_load_dir)
            files.sort()
            path = f'{auto_load_dir}/{files[-1]}'
        with open(path, 'rb') as f:
            sav = torch.load(f)
        net = sav['net']
        ns = NetShell(net)

        state = sav.get('state')
        if state:
            ns.state = state
        return ns

    def __init__(self, net, Dataset=None, criterion=torch.nn.CrossEntropyLoss()):
        self.net = net
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.criterion = criterion
        self.path = {
            'dataset': None,
            'dataset_root': 'cookie/data',
            'save': 'cookie/save',
        }
        for fp in self.path.values():
            if fp and not os.path.exists(fp):
                os.makedirs(fp)

        self.transform = {
            'train': NetShell.default_transform,
            'test': NetShell.default_transform,
        }
        self.dataset = {
            'train': None,
            'test': None,
        }
        if Dataset:
            self.load_dataset(Dataset)
        else:
            self.dataset['train'] = torchvision.datasets.CIFAR10(root=self.path['dataset_root'],
                                                                 transform=self.transform['train'], download=True)
            self.dataset['test'] = torchvision.datasets.CIFAR10(root=self.path['dataset_root'],
                                                                transform=self.transform['test'], download=True)

        self.shuffle = {
            'train': False,
            'test': False,
        }
        self.state = {
            'epoch': 0,
            'iter': 0,
        }
        self.max_epoch = 200
        self.cuda = True
        self.save_every = {
            'checkpoint': False,
            'epoch': 1,
        }
        self.early_stop_max = 5

    def train(self, batch_size=8, num_workers=2, checkpoint=2000, early_stop=False, sampling_test=True):
        trainloader = torch.utils.data.DataLoader(self.dataset['train'], batch_size=batch_size,
                                                  shuffle=self.shuffle['train'], num_workers=num_workers)
        base_epoch = self.state['epoch']
        cuda = self.cuda
        net = self.net
        optimizer = self.optimizer
        criterion = self.criterion
        checkpoint_timer = 0
        if cuda:
            net.cuda()
        early_stop_flag = 0
        early_stop_last_loss = 0
        for epoch in range(base_epoch, self.max_epoch):
            then = datetime.now()
            running_loss = 0
            total_itr = 0
            for itr, data in enumerate(trainloader, 0):
                xs, truth = data
                if cuda:
                    xs, truth = Variable(xs.cuda()), Variable(truth.cuda())
                else:
                    xs, truth = Variable(xs), Variable(truth)
                optimizer.zero_grad()
                ys = net(xs)
                loss = criterion(ys, truth)
                loss.backward()
                optimizer.step()
                running_loss += loss.data.item()
                if itr != 0 and itr % checkpoint == 0:
                    now = datetime.now()
                    average_loss = running_loss / checkpoint
                    print(f'[{epoch + 1}, {itr:5d}] loss:{average_loss:.10f} | {(now - then).seconds}s')
                    running_loss = 0.0
                    then = now
                    checkpoint_timer += 1
                    if self.save_every['checkpoint'] and checkpoint_timer % self.save_every['checkpoint'] == 0:
                        self.state['iter'] = itr
                        self.save()
                    if early_stop:
                        if average_loss >= early_stop_last_loss:
                            early_stop_flag += 1
                            if early_stop_flag > self.early_stop_max:
                                return
                        else:
                            early_stop_flag = 0
                        early_stop_last_loss = average_loss
                total_itr = itr
            self.state['iter'] = total_itr
            self.state['epoch'] = epoch + 1
            if self.save_every['epoch'] and (epoch + 1) % self.save_every['epoch'] == 0:
                self.save()
            if sampling_test:
                self.test(batch_size=8, num_workers=2)

    def test(self, batch_size=8, num_workers=2):
        testloader = torch.utils.data.DataLoader(self.dataset['test'], batch_size=batch_size,
                                                 shuffle=self.shuffle['test'], num_workers=num_workers)
        net = self.net
        cuda = self.cuda
        if cuda:
            net.cuda()
        correct = 0
        total = 0
        for data in testloader:
            xs, truth = data
            if cuda:
                xs, truth = Variable(xs.cuda()), Variable(truth.cuda())
            else:
                xs, truth = Variable(xs), Variable(truth)
            ys = net(xs)
            _, predicted = torch.max(ys.data, 1)
            total += truth.size(0)
            correct += (predicted == truth).sum()
        print('Accuracy of the network on the test dataset: %d %%' % (100 * correct / total))
        return 100 * correct / total

    def load_dataset(self, Dataset):
        pass

    def save(self, save_optimizer=False, save_state=True):
        pkg = {'net': self.net}
        suffix = 'n'
        if save_optimizer:
            pkg['optimizer'] = self.optimizer.state_dict()
            suffix += 'o'
        if save_state:
            pkg['state'] = self.state
            suffix += 's'
        suffix += '.sav'
        nhash = self.net_hash()
        fp = f"{self.path['save']}/{nhash}.{suffix}"
        torch.save(pkg, fp)
        print(f'saved as {fp}')

    def net_hash(self, special_mark=None):
        if special_mark:
            special_mark = f'x{special_mark}'
        else:
            special_mark = ''
        return f"{self.net._get_name()}{special_mark}_ep{self.state['epoch']}_{self.state['iter']}"


if __name__ == '__main__':
    net = LeNet()
    ns = NetShell(net)
    ns.save_every['epoch'] = 10
    ns.train(batch_size=64, checkpoint=400, early_stop=True)

    # ns = NetShell.sav_loader(auto_load_dir='cookie/save/')
    # ns.train(batch_size=64, checkpoint=400)
