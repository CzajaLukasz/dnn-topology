import os
import argparse
from config import NPROC, MAX_EPSILON, UPPER_DIM
import time

def log_timing(message, args):
    with open("timing_logs.txt", "a") as f:
        f.write(f"[TOPOLOGY | {args.net}_{args.dataset}] {message}\n")

parser = argparse.ArgumentParser()
parser.add_argument('--save_path')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0)
parser.add_argument('--epochs', nargs='+')
parser.add_argument('--thresholds', nargs='+', type=float)
args = parser.parse_args()

path = os.path.join(args.save_path, args.net+"_"+args.dataset+"/")

for e in args.epochs:
    t_epoch_dipha = time.time()

    t_sparse = time.time()
    os.system("/opt/dipha/build/full_to_sparse_distance_matrix "+str(MAX_EPSILON)+" "+path+"adj_epc{}_trl{}.bin ".format(e, args.trial)+
              path+"adj_epc{}_trl{}_{}.bin".format(e, args.trial, MAX_EPSILON))
    log_timing(f"Epoka {e} - Konwersja na format rzadki: {time.time() - t_sparse:.2f} s", args)

    t_mpi = time.time()
    os.system("mpiexec -n "+str(NPROC)+" /opt/dipha/build/dipha --upper_dim "+str(UPPER_DIM)+" --benchmark  --dual "+path+
              "adj_epc{}_trl{}_{}.bin ".format(e, args.trial, MAX_EPSILON)+path+"adj_epc{}_trl{}_{}.bin.out".format( e, args.trial, MAX_EPSILON))
    log_timing(f"Epoka {e} - Obliczenia DIPHA (MPI): {time.time() - t_mpi:.2f} s", args) # <--- ZAPIS
    
    log_timing(f"Epoka {e} - CAŁKOWITY CZAS TOPOLOGII: {time.time() - t_epoch_dipha:.2f} s", args)
