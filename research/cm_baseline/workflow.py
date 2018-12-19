import torchvision

from research.cm_baseline.AGG4k import AGG4k
from shell import NetShell

if __name__ == '__main__':
    # net = torchvision.models.resnet34(pretrained=True)
    # ns = NetShell(net, AGG4k, save_path='cookie/save/resnet-34')

    ns = NetShell.sav_loader(auto_load_dir='cookie/save/resnet-34', Dataset=AGG4k, use_optimizer=False)
    ns.shuffle['train'] = True
    ns.shuffle['test'] = True
    ns.save_every['epoch'] = 1
    ns.train(batch_size=64, checkpoint=400)
