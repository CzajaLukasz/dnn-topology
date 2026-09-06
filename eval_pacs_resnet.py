import os
import glob
import pickle
import struct
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EPOCHS = [1, 10, 25, 50, 75, 100]
DOMAINS_TEST = ["art", "cartoon", "sketch"]


def read_dipha_diagram(file_path):
    """
    Dekoduje binarny plik wyjściowy DIPHA (.bin.out).
    Zwraca tablicę punktów: [wymiar_homologii, birth, death].
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            magic = struct.unpack("<q", f.read(8))[0]
            if magic != 8067171840:
                return None
            
            file_type = struct.unpack("<q", f.read(8))[0]
            if file_type != 2:
                return None
                
            num_pts = struct.unpack("<q", f.read(8))[0]
            
            pts = []
            for _ in range(num_pts):
                dim = struct.unpack("<q", f.read(8))[0]
                birth = struct.unpack("<d", f.read(8))[0]
                death = struct.unpack("<d", f.read(8))[0]
                pts.append([dim, birth, death])
                
        return np.array(pts) if pts else np.empty((0, 3))
    except Exception:
        return None


def extract_tda_features(pts):
    """
    Ekstrakcja 8 deskryptorów diagramu persistencji dla grup H0 i H1.
    """
    if pts is None or len(pts) == 0:
        return np.zeros(8)

    h0 = pts[pts[:, 0] == 0]
    h1 = pts[pts[:, 0] == 1]

    def get_dim_features(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0, 0.0
        births = arr[:, 1]
        deaths = arr[:, 2]
        
        valid = np.isfinite(deaths) & np.isfinite(births)
        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0.0
            
        lifetimes = np.clip(deaths[valid] - births[valid], a_min=0.0, a_max=None)
        if len(lifetimes) == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        tot_pers = float(np.sum(lifetimes))
        mean_pers = float(np.mean(lifetimes))
        max_pers = float(np.max(lifetimes))
        
        if tot_pers > 1e-9:
            p = lifetimes / tot_pers
            p = p[p > 0]
            entropy = -float(np.sum(p * np.log(p)))
        else:
            entropy = 0.0
            
        return tot_pers, mean_pers, max_pers, entropy

    h0_feats = get_dim_features(h0)
    h1_feats = get_dim_features(h1)

    return np.array(list(h0_feats) + list(h1_feats))


def load_gaps_from_stats(stats_path):
    """
    Wczytuje luki generalizacji z listy słowników w pliku .pkl.
    Luka = (acc_tr - acc_te) / 100.0
    """
    if not os.path.exists(stats_path):
        return {}

    try:
        with open(stats_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return {}

    gaps = {}
    if isinstance(data, list):
        for ep in EPOCHS:
            idx = ep - 1
            if idx < len(data):
                entry = data[idx]
                if isinstance(entry, dict) and "acc_tr" in entry and "acc_te" in entry:
                    acc_tr = float(entry["acc_tr"])
                    acc_te = float(entry["acc_te"])
                    gaps[ep] = (acc_tr - acc_te) / 100.0
    return gaps


def load_domain_data(domain, trials, net="convnext_tiny"):
    X_list = []
    y_list = []

    target_dir = f"pacs_{domain}"
    cand_losses = glob.glob(f"./losses/{net}_pacs_{domain}*")
    cand_ckpts = glob.glob(f"./checkpoint/{net}_pacs_{domain}*")

    loss_base = cand_losses[0] if cand_losses else f"./losses/{net}_{target_dir}"
    ckpt_base = cand_ckpts[0] if cand_ckpts else f"./checkpoint/{net}_{target_dir}"

    for trial in trials:
        stats_path = os.path.join(loss_base, f"stats_trial_{trial}.pkl")
        gaps_dict = load_gaps_from_stats(stats_path)

        for ep in EPOCHS:
            pattern = os.path.join(ckpt_base, f"adj_epc{ep}_trl{trial}_*.bin.out")
            matches = glob.glob(pattern)
            
            if not matches:
                continue

            diag_file = matches[0]
            pts = read_dipha_diagram(diag_file)
            feats = extract_tda_features(pts)
            
            gap = gaps_dict.get(ep, np.nan)
            if not np.isnan(gap):
                X_list.append(feats)
                y_list.append(gap)

    return np.array(X_list), np.array(y_list)


def evaluate_predictions(y_true, y_pred, name=""):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9 and len(y_true) > 2:
        r_val, r_p = pearsonr(y_true, y_pred)
        rho_val, rho_p = spearmanr(y_true, y_pred)
        tau_val, tau_p = kendalltau(y_true, y_pred)
    else:
        r_val, r_p = np.nan, np.nan
        rho_val, rho_p = np.nan, np.nan
        tau_val, tau_p = np.nan, np.nan

    print(f"\n==================== {name} ====================")
    print(f"R^2 Score:         {r2:.4f}")
    print(f"MAE:               {mae:.4f}")
    print(f"RMSE:              {rmse:.4f}")
    print(f"Pearson r:         {r_val:.4f} (p = {r_p:.4e})")
    print(f"Spearman rho:      {rho_val:.4f} (p = {rho_p:.4e})")
    print(f"Kendall tau:       {tau_val:.4f} (p = {tau_p:.4e})")


def main():
    print("Ładowanie danych z domeny źródłowej (PACS Photo - Triale 1-6)...")
    X_train, y_train = load_domain_data("photo", trials=[1, 2, 3, 4, 5, 6])
    
    if len(X_train) == 0:
        print("Nie udało się załadować danych dla PACS Photo!")
        return

    print(f"Wczytano {len(X_train)} próbek (6 triali x {len(EPOCHS)} epok).")

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    evaluate_predictions(y_train, y_pred_train, name="PACS Photo (Domenowe źródło - Trening)")

    for domain in DOMAINS_TEST:
        X_ood, y_ood = load_domain_data(domain, trials=[1])
        if len(X_ood) == 0:
            print(f"\nBrak kompletnych danych dla domeny docelowej: {domain}")
            continue
            
        y_pred_ood = model.predict(X_ood)
        evaluate_predictions(y_ood, y_pred_ood, name=f"OOD Transfer: Photo -> {domain.capitalize()}")


if __name__ == "__main__":
    main()