import torchvision

from research.cm_baseline.AGG4k import AGG4k
from shell import NetShell

if __name__ == '__main__':
    ns = NetShell.sav_loader(auto_load_path='../../cookie/save/resnet-34', Dataset=AGG4k, load_optimizer=False)
    ns.path.dataset_dir = 'AGG4k'
    ns.path.save_dir = 'resnet-34'
    op = ns.options
    op.train.shuffle = True
    op.train.save_epoch = 2
    op.test.shuffle = True
    ns.train(batch_size=32, checkpoint=20)
