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

# Ustal optymalną liczbę procesów dla DIPHA (8 lub 16 jest najszybsze)
#NPROC = min(int(NPROC), 16)

for e in args.epochs:
    t_epoch_dipha = time.time()

    # 1. Konwersja na format rzadki
    t_sparse = time.time()
    sparse_cmd = (
        f"/opt/dipha/build/full_to_sparse_distance_matrix {MAX_EPSILON} "
        f"{path}adj_epc{e}_trl{args.trial}.bin "
        f"{path}adj_epc{e}_trl{args.trial}_{MAX_EPSILON}.bin"
    )
    os.system(sparse_cmd)
    log_timing(f"Epoka {e} - Konwersja na format rzadki: {time.time() - t_sparse:.2f} s", args)

    # 2. Obliczenia DIPHA z flagami optymalizacyjnymi OpenMPI
    t_mpi = time.time()
    mpi_cmd = (
        f"mpiexec --allow-run-as-root --oversubscribe "
        f"--mca btl_vader_single_copy_mechanism none "
        f"-n {NPROC} "
        f"/opt/dipha/build/dipha --upper_dim {UPPER_DIM} --benchmark --dual "
        f"{path}adj_epc{e}_trl{args.trial}_{MAX_EPSILON}.bin "
        f"{path}adj_epc{e}_trl{args.trial}_{MAX_EPSILON}.bin.out"
    )
    os.system(mpi_cmd)
    log_timing(f"Epoka {e} - Obliczenia DIPHA (MPI, {NPROC} procs): {time.time() - t_mpi:.2f} s", args)
    
    log_timing(f"Epoka {e} - CAŁKOWITY CZAS TOPOLOGII: {time.time() - t_epoch_dipha:.2f} s", args)