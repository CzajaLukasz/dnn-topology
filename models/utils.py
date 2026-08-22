from .lenet import *
from .vgg import *
from .resnet import *
from .alexnet import *
from .densenet import *
from .inception import *
from .conv_x import * 
from .convnext import *


num_classes = {
    'mnist': 10,
    'usps': 10,
    'cifar10': 10,
    'cifar10_gray': 10, 
    'mnist_adversarial': 10,
    'imagenet': 200,
    'imagenet_gray': 200,
    'vgg_cifar10_adversarial': 10
}


def get_model(name, dataset):
    net = None

    if name == 'conv_2':
        net = Conv_2(num_classes=10)
    elif name == 'conv_4':
        net = Conv_4(num_classes=10)
    elif name == 'conv_6':
        net = Conv_6(num_classes=10)
    elif name == 'lenet_300_100' and dataset in ['mnist', 'usps', 'cifar10_gray28', 'fashion_mnist', 'svhn_gray28']:
        net = LeNet_300_100(num_classes=10)
    elif name == 'lenet' and dataset in ['mnist', 'usps', 'cifar10_gray28', 'mnist_adversarial', 'fashion_mnist', 'svhn_gray28']:
        net = LeNet(num_classes=10)
    elif name == 'lenet' and dataset == 'imagenet':
        net = LeNet(num_classes=200)
    elif name == 'lenetbin':
        net = LeNet(num_classes=2)
    elif name == 'lenet32bin':
        net = LeNet(num_classes=2, input_size=32)
    elif name == 'lenet32' and dataset in ['mnist', 'usps', 'cifar10_gray', 'mnist_adversarial', 'svhn']:
        net = LeNet(num_classes=10, input_size=32)
    elif name == 'lenet32' and dataset == 'imagenet_gray':
        net = LeNet(num_classes=200, input_size=32)
    elif name == 'lenetext' and dataset in ['mnist', 'usps']:
        net = LeNetExt(n_channels=1, num_classes=10)
    elif name == 'lenetext' and dataset == 'cifar10':
        net = LeNetExt(n_channels=3, num_classes=10)
    elif name == 'vgg' and dataset in ['cifar10', 'mnist', 'usps', 'mnist_color32', 'fashion_mnist_color32', 'vgg_cifar10_adversarial', 'svhn']:
        net = VGG('VGG16', num_classes=10)
    elif name == 'vgg' and dataset == 'imagenet':
        net = VGG('VGG16', num_classes=200)
    # Obsługa zarówno nazwy 'resnet', jak i 'resnet18'
    elif name in ['resnet', 'resnet18'] and dataset in ['cifar10', 'mnist', 'usps', 'mnist_color32', 'fashion_mnist_color32', 'svhn']:
        net = ResNet18(num_classes=10)
    elif name in ['resnet', 'resnet18'] and dataset == 'imagenet':
        net = ResNet18(num_classes=200)
    elif name == 'densenet' and dataset == 'cifar10':
        net = DenseNet121(num_classes=10)    
    elif name == 'densenet' and dataset == 'imagenet':
        net = DenseNet121(num_classes=200)
    elif name == 'inception' and dataset == 'cifar10':
        net = GoogLeNet(num_classes=10)
    elif name == 'inception' and dataset == 'imagenet':
        net = GoogLeNet(num_classes=200)
    elif name == 'alexnet' and dataset in ['cifar10', 'mnist', 'usps']:
        net = AlexNet(num_classes=10)
    elif name == 'alexnet' and dataset == 'imagenet':
        net = AlexNet(num_classes=200)
    elif name == 'convnext_tiny' and dataset in ['cifar10', 'cifar10_convnext', 'mnist', 'usps']:
        net = TopologicalConvNeXt(num_classes=10)
    elif name == 'convnext_tiny' and dataset == 'imagenet':
        net = TopologicalConvNeXt(num_classes=200)

    # Bezpiecznik chroniący przed UnboundLocalError
    if net is None:
        raise ValueError(f"Nieobsługiwana kombinacja argumentów: name='{name}', dataset='{dataset}'")

    return net


def get_criterion(dataset):
    criterion = nn.CrossEntropyLoss()
    return criterion 


def init_from_checkpoint(net, args):
    ''' Initialize from checkpoint'''
    print('==> Initializing from fixed checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.net + '_' +args.dataset + '/ckpt_trial_' + str(args.fixed_init) + '_epoch_50.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net, best_acc, start_epoch