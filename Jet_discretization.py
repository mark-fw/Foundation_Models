import numpy as np
import pandas as pd


def discretize_jet_features_pT_only(
    df,
    pT_prefix='pT_',
    eta_prefix='delta_eta_',
    phi_prefix='delta_phi_',
    n_particles=200,
    eta_phi_range=(-0.8, 0.8),
    n_eta_phi_bins=29,
    n_logpt_bins=39,
    logpt_min_percentile=0.1,   # 0.1 -> 0.1th percentile (≈99.9% darüber)
    nan_to_underflow=False,      # True -> NaNs werden als underflow (0) kodiert; False -> NaN -> -1
    # Neue Optionen für vorgegebene log-pT-Kanten:
    logpt_edges=None,           # array-like of length n_logpt_bins+1 -> use these edges exactly (for val/test)
    logpt_min=None,             # float: if provided (with logpt_max) use to construct edges
    logpt_max=None,
):
    """
    Diskretisiert pT_i, eta_i, phi_i für i=0..(n_particles-1).
    Erwartet exakt die Spalten:
      pT_0 ... pT_{n_particles-1},
      eta_0 ... eta_{n_particles-1},
      phi_0 ... phi_{n_particles-1}.
    Rückgabe: (out_df, edges_dict)
      out_df: Spalten 'pt_bin_i', 'eta_bin_i', 'phi_bin_i' für i=0..n_particles-1
      edges_dict: {'eta_phi_edges', 'logpt_edges', 'logpt_min','logpt_max'}

    Binning-Übersicht (wie im Paper):
      - eta,phi: 29 zentrale Bins in eta_phi_range plus underflow bin 0 und overflow bin n_eta_phi_bins+1
                 -> result indices in [0 .. n_eta_phi_bins+1] (standard: 0..30)
      - pT (log): n_logpt_bins zentrale Bins in [logpt_min, logpt_max] plus underflow 0 and overflow n_logpt_bins+1
                 -> result indices in [0 .. n_logpt_bins+1] (standard: 0..40)

    Besonderheit:
      - Wenn `logpt_edges` übergeben wird, werden diese exakt verwendet (d.h. keine Berechnung aus Daten).
      - Wenn `logpt_min` und `logpt_max` übergeben werden, werden äquidistante edges erzeugt.
      - Sonst: logpt_min aus Daten mittels Percentile (logpt_min_percentile).
    """
    # --- 1) prüfe Spalten ---
    expected_pt_cols = [f'{pT_prefix}{i}' for i in range(n_particles)]
    expected_eta_cols = [f'{eta_prefix}{i}' for i in range(n_particles)]
    expected_phi_cols = [f'{phi_prefix}{i}' for i in range(n_particles)]

    missing = []
    for col in expected_pt_cols + expected_eta_cols + expected_phi_cols:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Es fehlen erwartete Spalten ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")

    # --- 2) Arrays extrahieren ---
    pt_arr = df[expected_pt_cols].to_numpy(dtype=float)   # shape (n_events, n_particles)
    eta_arr = df[expected_eta_cols].to_numpy(dtype=float)
    phi_arr = df[expected_phi_cols].to_numpy(dtype=float)
    n_events = pt_arr.shape[0]

    # --- 3) ETA/PHI Bins (fest wie zuvor) ---
    a, b = eta_phi_range
    edges_eta = np.linspace(a, b, n_eta_phi_bins + 1)   # length n_eta_phi_bins+1

    eta_idx = np.searchsorted(edges_eta, eta_arr, side='right') - 1
    phi_idx = np.searchsorted(edges_eta, phi_arr, side='right') - 1

    eta_idx_mapped = np.empty_like(eta_idx, dtype=int)
    phi_idx_mapped = np.empty_like(phi_idx, dtype=int)

    # underflow
    eta_idx_mapped[eta_idx == -1] = 0
    phi_idx_mapped[phi_idx == -1] = 0
    # overflow
    eta_idx_mapped[eta_idx == n_eta_phi_bins] = n_eta_phi_bins + 1
    phi_idx_mapped[phi_idx == n_eta_phi_bins] = n_eta_phi_bins + 1
    # in-range -> +1
    inrange_eta = (eta_idx >= 0) & (eta_idx < n_eta_phi_bins)
    inrange_phi = (phi_idx >= 0) & (phi_idx < n_eta_phi_bins)
    eta_idx_mapped[inrange_eta] = eta_idx[inrange_eta] + 1
    phi_idx_mapped[inrange_phi] = phi_idx[ inrange_phi] + 1

    if not nan_to_underflow:
        eta_idx_mapped[np.isnan(eta_arr)] = -1
        phi_idx_mapped[np.isnan(phi_arr)] = -1

    # --- 4) LOG pT Bins (jetzt mit optionaler Vorgabe) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        logpt_arr = np.where(pt_arr > 0, np.log(pt_arr), np.nan)  # NaN für pT<=0 oder NaN

    # Wenn logpt_edges vorgegeben -> verwende diese unverändert
    if logpt_edges is not None:
        edges_logpt = np.asarray(logpt_edges, dtype=float)
        if edges_logpt.ndim != 1:
            raise ValueError("logpt_edges muss 1D-array-like sein.")
        if edges_logpt.size != n_logpt_bins + 1:
            raise ValueError(f"logpt_edges muss Länge {n_logpt_bins+1} haben (got {edges_logpt.size}).")
        # Monotonie prüfen
        if not np.all(np.diff(edges_logpt) > 0):
            raise ValueError("logpt_edges muss streng monoton wachsend sein.")
        logpt_min_used = float(edges_logpt[0])
        logpt_max_used = float(edges_logpt[-1])
    else:
        # Falls logpt_min und logpt_max explizit übergeben wurden -> verwende diese
        if (logpt_min is not None) and (logpt_max is not None):
            logpt_min_used = float(logpt_min)
            logpt_max_used = float(logpt_max)
        else:
            # Berechne aus den (positiven) pT-Werten im aktuellen df
            all_logpt = logpt_arr.flatten()
            all_logpt = all_logpt[np.isfinite(all_logpt)]
            if all_logpt.size == 0:
                raise ValueError("Keine positiven pT-Werte zum Bestimmen des log(pT)-Binnings gefunden.")
            logpt_min_used = float(np.percentile(all_logpt, logpt_min_percentile))
            logpt_max_used = float(np.max(all_logpt))
        # falls Min==Max, weiche leicht aus
        if np.isclose(logpt_min_used, logpt_max_used):
            logpt_max_used = logpt_min_used + 1e-6
        edges_logpt = np.linspace(logpt_min_used, logpt_max_used, n_logpt_bins + 1)

    # mappe logpt zu bins analog zu vorhin:
    logpt_idx = np.searchsorted(edges_logpt, logpt_arr, side='right') - 1  # -1 .. n_logpt_bins
    pt_idx_mapped = np.empty_like(logpt_idx, dtype=int)
    pt_idx_mapped[logpt_idx == -1] = 0
    pt_idx_mapped[logpt_idx == n_logpt_bins] = n_logpt_bins + 1
    inrange_pt = (logpt_idx >= 0) & (logpt_idx < n_logpt_bins)
    pt_idx_mapped[inrange_pt] = logpt_idx[inrange_pt] + 1

    if not nan_to_underflow:
        pt_idx_mapped[np.isnan(logpt_arr)] = -1

    # --- 5) Baue Ausgabedatenframe ---
    cols = []
    arrays = []

    for i in range(n_particles):
        arrays.append(pt_idx_mapped[:, i])
        cols.append(f'pT_bin_{i}')
        arrays.append(eta_idx_mapped[:, i])
        cols.append(f'eta_bin_{i}')
        arrays.append(phi_idx_mapped[:, i])
        cols.append(f'phi_bin_{i}')

    out = pd.DataFrame(
        np.column_stack(arrays),
        columns=cols,
        index=df.index
    )

    out["is_signal_new"] = df["is_signal_new"].values

    edges = {
        'eta_phi_edges': edges_eta,
        'logpT_edges': edges_logpt,
        'logpT_min': float(edges_logpt[0]),
        'logpT_max': float(edges_logpt[-1]),
    }

    return out, edges

data_train_0 = pd.read_hdf("/hpcwork/thes1906/Foundation_Models/Top_Quark_Tagging_Ref_Data/train_pT_eta_phi.h5", key="df")
data_val_0 = pd.read_hdf("/hpcwork/thes1906/Foundation_Models/Top_Quark_Tagging_Ref_Data/val_pT_eta_phi.h5", key="df")
data_test_0 = pd.read_hdf("/hpcwork/thes1906/Foundation_Models/Top_Quark_Tagging_Ref_Data/test_pT_eta_phi.h5", key="df")

data_type = "Top"

if data_type == "QCD":
    n = 0
elif data_type == "Top":
    n = 1

data_train_1 = data_train_0[data_train_0["is_signal_new"]==n].copy()
data_val_1 = data_val_0[data_val_0["is_signal_new"]==n].copy()
data_test_1 = data_test_0[data_test_0["is_signal_new"]==n].copy()

bins_train, edges = discretize_jet_features_pT_only(data_train_1)
edges_df = pd.DataFrame({'eta_phi_egdes': [edges["eta_phi_edges"]], 'logpT_egdes': [edges["logpT_edges"]]})
edges_df.to_hdf(f"/home/ew640340/Ph.D./Foundation_Models/New/{data_type}_bin_edges.h5", key="df")
bins_train.to_hdf(f"/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/{data_type}_train_discrete_pT_eta_phi.h5", key="df", mode="w")

bins_val, _ = discretize_jet_features_pT_only(data_val_1, logpt_edges=edges['logpT_edges'], logpt_min=edges['logpT_min'], logpt_max=edges['logpT_max'])
bins_val.to_hdf(f"/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/{data_type}_val_discrete_pT_eta_phi.h5", key="df", mode="w")

bins_test, _ = discretize_jet_features_pT_only(data_test_1, logpt_edges=edges['logpT_edges'], logpt_min=edges['logpT_min'], logpt_max=edges['logpT_max'])
bins_test.to_hdf(f"/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/{data_type}_test_discrete_pT_eta_phi.h5", key="df", mode="w")



