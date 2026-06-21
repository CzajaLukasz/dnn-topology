import argparse
import os
from config import SAVE_PATH
import time

def log_timing(message, args):
    with open("timing_logs.txt", "a") as f:
        # Dodajemy informacje o modelu, datasecie i trialu
        f.write(f"[MAIN | {args.net}_{args.dataset} | trial:{args.trial} | epochs_num:{args.n_epochs_train}] {message}\n")

parser = argparse.ArgumentParser()
parser.add_argument('--train', default=1, type=int)
parser.add_argument('--build_graph', default=1, type=int)
parser.add_argument('--net', help='Specify deep network architecture (e.g. lenet, alexnet, resnet, inception, vgg, etc)')
parser.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)')
parser.add_argument('--trial', default=0, help='Specify trial number. Used to differentiate btw multiple trainings of same setup.')
parser.add_argument('--n_epochs_train', default='5', help='Number of epochs to train.')
parser.add_argument('--lr', default='0.01', help='Specify learnig rate for training.')
parser.add_argument('--permute_labels', default='0.0', help='Specify if labels are going to be permuted. Float between 0 and 1. If 0, no permutation. If 1 all labels are permuted. Otherwise proportion of labels.')
parser.add_argument('--binarize_labels', default='-1', help='If positive, Binarize labels. Put label equal to binarize_labels to 1. All the rest put to zero.')
parser.add_argument('--data_subset', default='1.0', help='Specify if subset of data should be loaded. Float between 0 and 1. If 0, all data, else proportion of data randomly sampled.')
parser.add_argument('--epochs_test', help='Epochs for which you want to build graph. String of positive natural numbers separated by spaces.')
parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start step stop \' ')
parser.add_argument('--filtration', default='nominal')
parser.add_argument('--scale', default='linear')
parser.add_argument('--split',  default='0', help='Split network into partitions.')
parser.add_argument('--subsplit',  default='0', help='Split each partition.')
parser.add_argument('--kl', default='0', help='TO ADD.')
parser.add_argument('--graph_type', default='functional')
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--select_nodes', default='0')
parser.add_argument('--partition', default='hardcoded')
parser.add_argument('--batch_size', type=int, default=128, help='Rozmiar batcha dla treningu')
args = parser.parse_args()


def visible_print(message):
    ''' Visible print'''
    print('')
    print(50*'-')
    print(message)
    print(50*'-')
    print('')

log_timing("--- ROZPOCZĘCIE NOWEGO URUCHOMIENIA ---", args)
    
if args.train:
    visible_print('Training network')
    t0_train = time.time()
    os.system('python ./train.py --net '+str(args.net)+' --dataset '+str(args.dataset)+' --trial '+str(args.trial)+' --epochs '+
              args.n_epochs_train+' --lr '+str(args.lr)+' --permute_labels '+str(args.permute_labels)+' --train_batch_size '+str(args.batch_size)+' --subset '+str(args.data_subset)+
              ' --binarize_labels '+str(args.binarize_labels))
    log_timing(f"Faza treningu zajęła: {time.time() - t0_train:.2f} s", args)
    log_timing(f"Trening jednej epoki zajal srednio: {(time.time() - t0_train)/int(args.n_epochs_train):.2f} s", args)

if args.build_graph:
    visible_print('Building '+str(args.graph_type)+' graph')
    t0_graph = time.time()
    if args.graph_type == 'functional':
        os.system('python ./build_graph_functional.py --save_path '+SAVE_PATH+' --net '+str(args.net)+' --dataset '+
                  args.dataset+' --trial '+str(args.trial)+' --epochs '+str(args.epochs_test)+' --filtration '+str(args.filtration)+
                  ' --split '+str(args.split)+' --kl '+str(args.kl)+' --permute_labels '+str(args.permute_labels)+' --binarize_labels '+str(args.binarize_labels))
    elif args.graph_type == 'functional_big_networks':
        os.system('python ./build_graph_functional_big_networks.py --save_path '+SAVE_PATH+' --net '+str(args.net)+
                  ' --dataset '+str(args.dataset)+' --trial '+str(args.trial)+' --epochs '+str(args.epochs_test)+' --filtration '+
                  args.filtration+' --split '+str(args.split)+' --kl '+str(args.kl)+' --permute_labels '+str(args.permute_labels)+
                  ' --binarize_labels '+str(args.binarize_labels)+' --partition '+str(args.partition))
    elif args.graph_type == 'functional_persample':
        os.system('python ./build_graph_functional_persample.py --save_path '+SAVE_PATH+' --net '+str(args.net)+' --dataset '+
                  args.dataset+' --trial '+str(args.trial)+' --epochs '+str(args.epochs_test)+' --filtration '+str(args.filtration)+' --permute_labels '+
                  args.permute_labels+' --binarize_labels '+str(args.binarize_labels)+' --n_samples '+str(args.n_samples)+' --select_nodes '+
                  args.select_nodes)
    elif args.graph_type == 'functional_big_networks_per_partition':
        os.system('python ./build_graph_functional_big_networks_per_partition.py --save_path '+SAVE_PATH+' --net '+
                  args.net+' --dataset '+str(args.dataset)+' --trial '+str(args.trial)+' --epochs '+str(args.epochs_test)+' --filtration '+
                  args.filtration+' --split '+str(args.split)+ ' --subsplit '+str(args.subsplit)+' --kl '+str(args.kl)+' --permute_labels '+
                  args.permute_labels+' --binarize_labels '+str(args.binarize_labels)+' --partition '+str(args.partition))
    log_timing(f"Faza budowania grafu zajęła: {time.time() - t0_graph:.2f} s", args) # <--- ZAPIS DO PLIKU

visible_print('Computing topology')
os.system('python ./compute_topology.py --save_path '+SAVE_PATH+' --net '+str(args.net)+' --dataset '+str(args.dataset)+' --epochs '+
          args.epochs_test+' --trial '+str(args.trial))
