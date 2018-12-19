import os
from datetime import datetime

import torch
import torchvision
from torch import optim
from torch.autograd import Variable
from torchvision.transforms import transforms

from models.LeNet import LeNet


class NetShell:
    default_scale = 224
    default_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    class _Path:
        def __init__(self):
            self.root = '../..'
            self.dataset_root = 'cookie/data'
            self.dataset_dir = 'CIFAR10'
            self.save_root = 'cookie/save'
            self.save_dir = None

        def dataset_path(self):
            dataset_path = f'{self.root}/{self.dataset_root}'
            if self.dataset_dir:
                dataset_path = dataset_path + '/' + self.dataset_dir
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            return dataset_path

        def save_path(self):
            save_path = f'{self.root}/{self.save_root}'
            if self.save_dir:
                save_path = save_path + '/' + self.save_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    class _Train:
        def __init__(self):
            self.transform = NetShell.default_transform
            self.shuffle = True
            self.max_epoch = 200
            self.save_checkpoint = -1
            self.save_epoch = -1
            self.early_stop_max = 5

    class _Test:
        def __init__(self):
            self.transform = NetShell.default_transform
            self.shuffle = True

    class _Save:
        def __init__(self):
            self.optimizer = True
            self.state = True

    class _Options:
        def __init__(self):
            self.train = NetShell._Train()
            self.test = NetShell._Test()
            self.save = NetShell._Save()

    def __init__(self, net, Dataset=None, criterion=torch.nn.CrossEntropyLoss()):
        self.net = net
        self.cuda = True
        self.options = NetShell._Options()
        self.path = NetShell._Path()
        self.criterion = criterion
        self.Dataset = Dataset if Dataset else torchvision.datasets.CIFAR10
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.state = {
            'epoch': 0,
            'iter': 0,
            'special_mark': None,
        }

    @staticmethod
    def sav_loader(path=None, auto_load_path=None, Dataset=None, load_optimizer=False):
        if auto_load_path:
            files = os.listdir(auto_load_path)
            files.sort()
            path = f'{auto_load_path}/{files[-1]}'
        with open(path, 'rb') as f:
            sav = torch.load(f)
        net = sav['net']
        ns = NetShell(net, Dataset=Dataset)
        state = sav.get('state')
        if state:
            ns.state = state
        optimizer = sav.get('optimizer')
        if load_optimizer and optimizer:
            ns.optimizer.load_state_dict(optimizer)
        return ns

    def train(self, batch_size=8, num_workers=2, checkpoint=2000, sampling_test=True, early_stop=False):
        if self.Dataset:
            dataset = self.Dataset(root=self.path.dataset_path(), transform=self.options.train.transform, train=True)
        else:
            dataset = torchvision.datasets.CIFAR10(root=self.path.dataset_path(),
                                                   transform=self.options.train.transform,
                                                   download=True)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=self.options.train.shuffle, num_workers=num_workers)
        base_epoch = self.state['epoch']
        cuda = self.cuda
        net = self.net
        optimizer = self.optimizer
        criterion = self.criterion
        checkpoint_timer = 0
        checkpoint_save = self.options.train.save_checkpoint
        epoch_save = self.options.train.save_epoch
        if cuda:
            net.cuda()
        es_history = 0
        es_flag = 0
        for epoch in range(base_epoch, self.options.train.max_epoch):
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
                    if checkpoint_save > 0 and checkpoint_timer % checkpoint_save == 0:
                        self.state['iter'] = itr
                        self.save()
                total_itr = itr
            self.state['iter'] = total_itr
            self.state['epoch'] = epoch + 1
            if epoch_save > 0 and (epoch + 1) % epoch_save == 0:
                self.save()
            if sampling_test:
                accuracy = self.test(batch_size=batch_size, num_workers=num_workers)
                if early_stop:
                    if accuracy <= es_history:
                        es_flag += 1
                        if es_flag >= self.options.train.early_stop_max:
                            return
                    else:
                        es_flag = 0
                    es_history = accuracy

    def test(self, batch_size=8, num_workers=2):
        if self.Dataset:
            dataset = self.Dataset(root=self.path.dataset_path(), transform=self.options.test.transform, train=False)
        else:
            dataset = torchvision.datasets.CIFAR10(root=self.path.dataset_root,
                                                   train=False,
                                                   transform=self.options.test.transform,
                                                   download=True)

        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=self.options.test.shuffle, num_workers=num_workers)
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
        acc = 100.0 * correct.float() / total
        print(f'Accuracy of the network on the test dataset_root: {acc:.2f} %')
        return acc

    def save(self):
        pkg = {'net': self.net}
        suffix = 'n'
        if self.options.save.optimizer:
            pkg['optimizer'] = self.optimizer.state_dict()
            suffix += 'o'
        if self.options.save.state:
            pkg['state'] = self.state
            suffix += 's'
        suffix += '.sav'
        nhash = self.net_hash()
        fp = f"{self.path.save_path()}/{nhash}.{suffix}"
        torch.save(pkg, fp)
        print(f'saved as {fp}')

    def net_hash(self):
        special_mark = self.state.get('special_mark')
        if special_mark:
            special_mark = f'x{special_mark}'
        else:
            special_mark = ''
        return f"{self.net._get_name()}{special_mark}_ep{self.state['epoch']}_{self.state['iter']}"
