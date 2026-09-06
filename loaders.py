'''
Data loading Utilities for preparing for various different datasets.
Includes: MNIST, USPS, CIFAR10, SVHN, FashionMNIST, PACS.
'''

import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler, SequentialSampler, RandomSampler

# --- TRANSFORMS ---

TRANSFORMS_TR = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_COLOR32 = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 32, 32).expand(3, -1, -1)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_COLOR32 = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 32, 32).expand(3, -1, -1)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_CIFAR10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_CIFAR10_GRAY28 = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_CIFAR10_GRAY28 = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_SVHN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_SVHN_GRAY28 = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_SVHN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_SVHN_GRAY28 = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_IMAGENET = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))       
])

TRANSFORMS_TE_IMAGENET = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

TRANSFORMS_MNIST_ADV = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

TRANSFORMS_MNIST_TR = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

TRANSFORMS_MNIST_TE = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

TRANSFORMS_USPS_TR = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

TRANSFORMS_USPS_TE = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

TRANSFORMS_TR_CIFAR10_CONVNEXT = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_CIFAR10_CONVNEXT = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_PACS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

TRANSFORMS_TE_PACS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# --- DATA LOADERS ---

def loader(data, batch_size, subset=[], sampling=-1):
    ''' Interface to the dataloader function '''
    if data == 'mnist_train':
        return dataloader('mnist', './data', train=True, transform=TRANSFORMS_MNIST_TR, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'mnist_test':
        return dataloader('mnist', './data', train=False, transform=TRANSFORMS_MNIST_TE, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'usps_train':
        return dataloader('usps', './data', train=True, transform=TRANSFORMS_USPS_TR, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'usps_test':
        return dataloader('usps', './data', train=False, transform=TRANSFORMS_USPS_TE, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'mnist_color32_train':
        return dataloader('mnist', './data', train=True, transform=TRANSFORMS_TR_COLOR32, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'mnist_color32_test':
        return dataloader('mnist', './data', train=False, transform=TRANSFORMS_TE_COLOR32, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'cifar10_train':
        return dataloader('cifar10', './data', train=True, transform=TRANSFORMS_TR_CIFAR10, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'cifar10_gray28_train':
        return dataloader('cifar10', './data', train=True, transform=TRANSFORMS_TR_CIFAR10_GRAY28, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'cifar10_test':
        return dataloader('cifar10', './data', train=False, transform=TRANSFORMS_TE_CIFAR10, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'cifar10_gray28_test':
        return dataloader('cifar10', './data', train=False, transform=TRANSFORMS_TE_CIFAR10_GRAY28, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'svhn_train':
        return dataloader('svhn', './data', train='train', transform=TRANSFORMS_TR_SVHN, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'svhn_test':
        return dataloader('svhn', './data', train='test', transform=TRANSFORMS_TE_SVHN, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'svhn_gray28_train':
        return dataloader('svhn', './data', train='train', transform=TRANSFORMS_TR_SVHN_GRAY28, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'svhn_gray28_test':
        return dataloader('svhn', './data', train='test', transform=TRANSFORMS_TE_SVHN_GRAY28, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'fashion_mnist_train':
        return dataloader('fashion_mnist', './data', train=True, transform=TRANSFORMS_TR, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'fashion_mnist_test':
        return dataloader('fashion_mnist', './data', train=False, transform=TRANSFORMS_TE, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'fashion_mnist_color32_train':
        return dataloader('fashion_mnist', './data', train=True, transform=TRANSFORMS_TR_COLOR32, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'fashion_mnist_color32_test':
        return dataloader('fashion_mnist', './data', train=False, transform=TRANSFORMS_TE_COLOR32, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'cifar10_convnext_train':
        return dataloader('cifar10', './data', train=True, transform=TRANSFORMS_TR_CIFAR10_CONVNEXT, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data == 'cifar10_convnext_test':
        return dataloader('cifar10', './data', train=False, transform=TRANSFORMS_TE_CIFAR10_CONVNEXT, batch_size=batch_size, sampling=sampling, num_workers=4, subset=subset)
    elif data.startswith('pacs_'):
        domain = data.split('_')[1]
        domain_map = {
            'photo': 'photo',
            'art': 'art_painting',
            'cartoon': 'cartoon',
            'sketch': 'sketch'
        }
        actual_domain = domain_map.get(domain, domain)
        is_train = data.endswith('_train')
        trans = TRANSFORMS_TR_PACS if is_train else TRANSFORMS_TE_PACS
        return dataloader(f'pacs_{actual_domain}', f'./data/pacs/{actual_domain}', 
                          train=is_train, transform=trans, batch_size=batch_size, 
                          sampling=sampling, num_workers=4, subset=subset)
    else:
        raise ValueError(f"Nieobsługiwany zbiór danych w loaderze: {data}")


def get_dataset(data, path, train, transform):
    ''' Return dataset instance '''
    if data == 'mnist':
        dataset = torchvision.datasets.MNIST(path, train=train, download=True, transform=transform)
    elif data == 'usps':
        dataset = torchvision.datasets.USPS(path, train=train, download=True, transform=transform)
    elif data == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(path, train=train, download=True, transform=transform)
    elif data == 'svhn':
        dataset = torchvision.datasets.SVHN(path, split='train' if train else 'test', download=True, transform=transform)
    elif data == 'fashion_mnist':
        dataset = torchvision.datasets.FashionMNIST(path, train=train, download=True, transform=transform)
    elif data.startswith('pacs_'):
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset


def dataloader(data, path, train, transform, batch_size, num_workers, subset=[], sampling=-1):
    dataset = get_dataset(data, path, train, transform)
        
    if subset:
        # Zabezpieczenie przed wyjściem indeksu poza zakres rozmiaru zbioru
        valid_subset = [i for i in subset if i < len(dataset)]
        dataset = torch.utils.data.Subset(dataset, valid_subset)

    if sampling == -1:
        sampler = SequentialSampler(dataset)
    elif sampling == -2:
        sampler = RandomSampler(dataset)
    else:
        sampler = BinarySampler(dataset, sampling)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)


class BinarySampler(Sampler):
    """One-vs-rest sampling where pivot indicates the target class """

    def __init__(self, dataset, pivot):
        self.dataset = dataset
        self.pivot_indices = self._get_pivot_indices(pivot)
        self.nonpivot_indices = self._get_nonpivot_indices()
        self.indices = self._get_indices()
    
    def _get_targets(self):
        return [x for (_, x) in self.dataset]

    def _get_pivot_indices(self, pivot):
        return [i for i, x in enumerate(self._get_targets()) if x == pivot]

    def _get_nonpivot_indices(self):
        return random.sample(list(set(np.arange(len(self.dataset))) - set(self.pivot_indices)), len(self.pivot_indices))

    def _get_indices(self):
        return self.pivot_indices + self.nonpivot_indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
        
    def __len__(self):
        return len(self.indices)