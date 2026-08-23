import os
import argparse
import torch
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser(description='Ewaluacja regresji TDA -> Generalization Gap')
parser.add_argument('--net', default='convnext_tiny', type=str, help='Nazwa sieci: resnet18 lub convnext_tiny')
args = parser.parse_args()

def read_dipha_diagram(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, dtype=np.int64, count=1)[0]
        dtype_code = np.fromfile(f, dtype=np.int64, count=1)[0]
        n_pts = np.fromfile(f, dtype=np.int64, count=1)[0]
        dims = np.fromfile(f, dtype=np.int64, count=n_pts)
        births = np.fromfile(f, dtype=np.float64, count=n_pts)
        deaths = np.fromfile(f, dtype=np.float64, count=n_pts)
    return dims, births, deaths

def extract_topological_features(dims, births, deaths):
    valid = deaths < np.inf
    dims, births, deaths = dims[valid], births[valid], deaths[valid]
    lifetimes = deaths - births
    midpoints = (births + deaths) / 2.0
    
    h0_idx, h1_idx = dims == 0, dims == 1
    h0_life = lifetimes[h0_idx] if np.sum(h0_idx) > 0 else np.array([0.0])
    h0_mid = midpoints[h0_idx] if np.sum(h0_idx) > 0 else np.array([0.0])
    h1_life = lifetimes[h1_idx] if np.sum(h1_idx) > 0 else np.array([0.0])
    h1_mid = midpoints[h1_idx] if np.sum(h1_idx) > 0 else np.array([0.0])
    
    feats = [
        np.sum(h0_life), np.mean(h0_life), np.std(h0_life), np.max(h0_life),
        np.sum(h0_mid), np.mean(h0_mid),
        np.sum(h1_life), np.mean(h1_life), np.std(h1_life), np.max(h1_life),
        np.sum(h1_mid), np.mean(h1_mid),
        np.sum(h0_idx), np.sum(h1_idx)
    ]
    return np.nan_to_num(feats)

def collect_dataset(net_name, dataset_name, trials, epochs, root_dir='./checkpoint/'):
    records = []
    dir_path = os.path.join(root_dir, f"{net_name}_{dataset_name}")
    for trl in trials:
        for epc in epochs:
            ckpt_path = os.path.join(dir_path, f"ckpt_trial_{trl}_epoch_{epc}.t7")
            diag_path = os.path.join(dir_path, f"adj_epc{epc}_trl{trl}_0.3.bin.out")
            if not (os.path.exists(ckpt_path) and os.path.exists(diag_path)):
                continue
            ckpt = torch.load(ckpt_path, map_location='cpu')
            dims, births, deaths = read_dipha_diagram(diag_path)
            feats = extract_topological_features(dims, births, deaths)
            records.append({
                'dataset': dataset_name,
                'trial': trl,
                'epoch': epc,
                'train_acc': ckpt['train_acc'],
                'test_acc': ckpt['test_acc'],
                'gen_gap': ckpt['gen_gap'],
                'features': feats
            })
    return records

def print_metrics_table(y_true, y_pred, split_name="DATASET"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)
    kendall_tau, kendall_p = stats.kendalltau(y_true, y_pred)
    
    print(f"\n{'='*25} WYNIKI: {split_name} {'='*25}")
    print(f"{'Metryka':<35} | {'Wartość':<15}")
    print(f"{'-'*53}")
    print(f"{'Współczynnik determinacji (R^2)':<35} | {r2:.4f}")
    print(f"{'Średni błąd bezwzględny (MAE)':<35} | {mae:.4f}")
    print(f"{'Pierwiastek błędu średniokw. (RMSE)':<35} | {rmse:.4f}")
    print(f"{'-'*53}")
    print(f"{'Współczynnik Pearsona (r)':<35} | {pearson_r:.4f}")
    print(f"{'Wartość p-value (Pearson)':<35} | {pearson_p:.4e}")
    print(f"{'-'*53}")
    print(f"{'Współczynnik Spearmana (rho)':<35} | {spearman_rho:.4f}")
    print(f"{'Wartość p-value (Spearman)':<35} | {spearman_p:.4e}")
    print(f"{'-'*53}")
    print(f"{'Współczynnik Kendalla (tau)':<35} | {kendall_tau:.4f}")
    print(f"{'Wartość p-value (Kendall)':<35} | {kendall_p:.4e}")
    print(f"{'='*53}\n")

# Zbieranie danych
print(f"Pobieranie danych dla modelu: {args.net}")
mnist_data = collect_dataset(args.net, 'mnist', trials=[1, 2, 3, 4, 5, 6], epochs=[1, 10, 25, 50, 75, 100])
usps_data = collect_dataset(args.net, 'usps', trials=[1], epochs=[1, 10, 25, 50, 75, 100])

print(f"Liczba próbek MNIST: {len(mnist_data)}")
print(f"Liczba próbek USPS:  {len(usps_data)}")

if len(mnist_data) == 0 or len(usps_data) == 0:
    print("Brak kompletnych plików checkpoint/diagram do ewaluacji.")
    exit(0)

X_train = np.array([d['features'] for d in mnist_data])
y_train = np.array([d['gen_gap'] for d in mnist_data])
X_test = np.array([d['features'] for d in usps_data])
y_test = np.array([d['gen_gap'] for d in usps_data])

mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0) + 1e-8
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

model = Ridge(alpha=1.0)
model.fit(X_train_norm, y_train)

y_train_pred = model.predict(X_train_norm)
y_test_pred = model.predict(X_test_norm)

print_metrics_table(y_train, y_train_pred, split_name=f"{args.net} - MNIST (Trening)")
print_metrics_table(y_test, y_test_pred, split_name=f"{args.net} - USPS (OOD Transfer)")