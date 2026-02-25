import numpy as np
import pandas as pd
import re


def preprocess_4_momenta_to_pT_eta_phi(Jet_data):

    df = Jet_data.copy()

    # --- 1) bestimme Partikel-Indizes automatisch ---
    px_cols = [c for c in df.columns if re.match(r'^PX_\d+$', c)]
    if len(px_cols) == 0:
        raise ValueError("Keine Spalten vom Format 'PX_i' gefunden.")
    indices = sorted(int(re.search(r'(\d+)', c).group(1)) for c in px_cols)

    # sicherstellen, dass für jedes index auch PY_ und PZ_ existieren
    missing = [i for i in indices if f'PY_{i}' not in df.columns or f'PZ_{i}' not in df.columns]
    if missing:
        raise ValueError(f"Fehlende PY_/PZ_-Spalten für Indizes: {missing}")


    # --- 2) Baue NumPy-Arrays (n_events, n_particles) ---
    idx_list = indices
    px = df[[f'PX_{i}' for i in idx_list]].to_numpy(dtype=float)
    py = df[[f'PY_{i}' for i in idx_list]].to_numpy(dtype=float)
    pz = df[[f'PZ_{i}' for i in idx_list]].to_numpy(dtype=float)

    p_abs = np.sqrt(px**2 + py**2 + pz**2)

    pT = np.sqrt(px**2 + py**2)  # shape (n_events, n_particles)

    # Anzahl Events und Partikel
    n_events, n_particles = px.shape


    # --- 3) Jet-Achse (Summenimpuls) und Einheitsvektor n ---
    jet_px = px.sum(axis=1)   # shape (n_events)
    jet_py = py.sum(axis=1)
    jet_pz = pz.sum(axis=1)
    jet_norm = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)  # Betrag des Jet-Impulses

    # Vermeide Division durch 0: setze bei jet_norm==0 Nan (später behandeln)
    eps = 1e-12
    jet_norm_safe = np.where(jet_norm > eps, jet_norm, np.nan)

    n_z = jet_pz / jet_norm_safe    # (n_events,)

    eta_jet = np.arctanh(n_z)  # eta des Jets
    eta_particle = np.arctanh(pz / np.where(p_abs > eps, p_abs, np.nan))  # eta der Teilchen relativ zur Kollisionsachse
    delta_eta = eta_particle - eta_jet[:, None]  # (n_events, n_particles)

    phi_jet = np.arctan2(jet_py, jet_px)
    phi_particle_0 = np.arctan2(py, px)

    phi_particle = np.where(np.abs(phi_particle_0) > eps, phi_particle_0, np.nan)

    delta_phi_0 = phi_particle - phi_jet[:, None]
    delta_phi = (delta_phi_0 + np.pi) % (2 * np.pi) - np.pi


    # --- 6) Sortierung nach pT (absteigend) pro Event ---
    # Ersetze NaN in pT temporär mit -inf, damit NaNs ans Ende sortieren
    pT_for_sort = np.nan_to_num(pT, nan=-np.inf)
    order = np.argsort(-pT_for_sort, axis=1)  # (n_events, n_particles) ; Indizes der sortierten Partikel (desc)

    rows = np.arange(n_events)[:, None]
    pT_sorted = pT[rows, order]
    delta_eta_sorted = delta_eta[rows, order]
    delta_phi_sorted = delta_phi[rows, order]

    # Falls sowohl eta als auch phi = NaN, setze pT ebenfalls auf NaN ---
    delta_eta_phi_both_nan = np.isnan(delta_eta_sorted) & np.isnan(delta_phi_sorted)
    # Setze nur dort pT auf NaN (egal ob pT vorher schon NaN oder nicht)
    pT_sorted[delta_eta_phi_both_nan] = np.nan


    # --- 7) Baue Ausgabedatenframe mit Spalten pt_0.., eta_0.., phi_0.. (0 = größter pT) ---
    cols = []
    arrays = []

    for i in range(n_particles):
        arrays.append(pT_sorted[:, i])
        cols.append(f'pT_{i}')
        arrays.append(delta_eta_sorted[:, i])
        cols.append(f'delta_eta_{i}')
        arrays.append(delta_phi_sorted[:, i])
        cols.append(f'delta_phi_{i}')

    out = pd.DataFrame(
        np.column_stack(arrays),
        columns=cols,
        index=df.index
    )
    
    out["is_signal_new"] = df["is_signal_new"].values

    return out

data_0 = pd.read_hdf("/hpcwork/thes1906/Foundation_Models/Top_Quark_Tagging_Ref_Data/test.h5", key="/table")

data_1 = preprocess_4_momenta_to_pT_eta_phi(data_0)

data_1.to_hdf("/hpcwork/thes1906/Foundation_Models/Top_Quark_Tagging_Ref_Data/test_pT_eta_phi.h5", key="df", mode="w")






