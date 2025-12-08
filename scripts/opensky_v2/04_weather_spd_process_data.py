# %%
import os
from torch import long
import yaml
from pathlib import Path
from fastmeteo.source import ArcoEra5
from node_fdm.data.meteo_and_parameters import process_files
from node_fdm.data.split import make_global_split_csv

from node_fdm.architectures.opensky_v2.flight_process import selected_param_config

# %%
cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]
process_dir = data_dir / cfg["paths"]["process_dir"]
os.makedirs(process_dir, exist_ok=True)

era5_cache_dir = data_dir / cfg["paths"]["era5_cache_dir"]
os.makedirs(era5_cache_dir, exist_ok=True)

era5_features = cfg["era5_features"]


os.makedirs(process_dir, exist_ok=True)


arco_grid = ArcoEra5(local_store=era5_cache_dir, features=era5_features)

for file in os.listdir(preprocess_dir):
    file_path = preprocess_dir / file
    print(file_path)
    process_files(arco_grid, file_path, process_dir, selected_param_config)

make_global_split_csv(process_dir)

# %%
import os
import yaml
from pathlib import Path
from fastmeteo.source import ArcoEra5
from node_fdm.data.meteo_and_parameters import (
    process_files,
    build_spd_and_vert_selected_from_segments,
)
from node_fdm.data.split import make_global_split_csv

import sys

print(str(Path.cwd()))

sys.path.append("/data/common/dataiku/config/projects/NODEFDM/lib/python/src/node_fdm")

from architectures.opensky_v2.flight_process import (
    selected_param_config,
)


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
print(data_dir)
preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]
process_dir = data_dir / cfg["paths"]["process_dir"]

era5_cache_dir = data_dir / cfg["paths"]["era5_cache_dir"]

era5_features = cfg["era5_features"]

# %%
import pandas as pd

for file in os.listdir(preprocess_dir):
    file_path = preprocess_dir / file
    print(file_path)
    df = pd.read_parquet(file_path)
    break
# %%

from node_fdm.data.meteo_and_parameters import (
    process_files,
    build_spd_and_vert_selected_from_segments,
    compute_tas,
    compute_mach_and_cas,
)


arco_grid = ArcoEra5(local_store=era5_cache_dir, features=era5_features)
df = arco_grid.interpolate(df)
df["TAS"] = compute_tas(df)
df["Mach"], df["CAS"] = compute_mach_and_cas(
    df["TAS"], df["altitude"], df["temperature"]
)

# %%
tasks = ((f_id, f) for f_id, f in df.groupby("flight_id"))
# %%
import matplotlib.pyplot as plt

i = 0

selected_param_config = {
    "mach": {
        "tol": 0.0005,
        "min_len": 120,
        "alt_threshold": 15000,
        "smooth_window": 30,
        "use_alt": True,
    },
    "cas": {
        "tol": 0.75,
        "min_len": 20,
        "use_alt": False,
        "smooth_window": 20,
        "smooth_method": "savgol",
    },
    "vz": {
        "tol": 25,
        "min_len": 30,
        "use_alt": False,
        "min_abs_value": 25,
        "smooth_window": 20,
        "smooth_method": "savgol",
    },
    "alt": {
        "tol": 25,
        "min_len": 5,
        "use_alt": False,
        "min_abs_value": 25,
        "smooth_window": 5,
        "smooth_method": "savgol",
    },
    "add_alt": True,
}
from node_fdm.utils.physics.constants import T0, p0, g, R, gamma_ratio, a0, ftmn, kt
import numpy as np


from node_fdm.data.meteo_and_parameters import (
    detect_constant_segments,
    add_segment_column,
)

hgd_cfg = {
    "tol": 0.5,
    "min_len": 5,
    "use_alt": False,
    "min_abs_value": 1.0,
    "smooth_window": 5,
    "smooth_method": "savgol",
}


for f_id, f in tasks:
    print(f_id)
    res = build_spd_and_vert_selected_from_segments(f, selected_param_config)
    hgd_segments, _ = detect_constant_segments(res, var_name="heading", **hgd_cfg)
    res = add_segment_column(res, hgd_segments, "hgd_sel", fill_value=np.nan)
    res.loc[res.index[-1], "hgd_sel"] = res.loc[res.index[-1], "heading"]
    res["hgd_sel"] = res["hgd_sel"].bfill()

    plt.plot(res["timestamp"], np.unwrap(res["track"]))
    plt.plot(res["timestamp"], np.unwrap(res["heading"]))
    # plt.plot(res["timestamp"], res["hgd_sel"])
    plt.show()
    i += 1
    if i == 5:
        break


# %%
res
# %%
import matplotlib.pyplot as plt

plt.plot(res["timestamp"], res["mach_sel"])
plt.plot(res["timestamp"], res["Mach"])
# %%
plt.plot(res["timestamp"], res["cas_sel"])
plt.plot(res["timestamp"], res["CAS"])


# %%
res.columns
# %%

import numpy as np
import matplotlib.pyplot as plt

# Rayon moyen de la Terre en kilomètres
EARTH_RADIUS_KM = 6371.0

# --- Fonctions Géodésiques ---


def to_radians(degrees):
    """Convertit degrés en radians."""
    return np.radians(degrees)


def to_degrees(radians):
    """Convertit radians en degrés."""
    return np.degrees(radians)


def calculate_initial_bearing(phi1, lambda1, phi2, lambda2):
    """Calcule l'azimut initial (Track A -> B) en radians."""
    d_lambda = lambda2 - lambda1
    y = np.sin(d_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lambda)
    return np.arctan2(y, x)


def calculate_current_bearing(phi1, lambda1, phi2, lambda2):
    """Calcule l'azimut courant entre un point C(phi1, lambda1) et B(phi2, lambda2) en radians."""
    d_lambda = lambda2 - lambda1
    y = np.sin(d_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lambda)
    return np.arctan2(y, x)


def great_circle_distance_km(phi1, lambda1, phi2, lambda2):
    """Calcule la distance de la Grande Orthodromie A -> B en km."""
    d_sigma = np.arccos(
        np.sin(phi1) * np.sin(phi2)
        + np.cos(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)
    )
    return d_sigma * EARTH_RADIUS_KM


def calculate_intermediate_point(phi1, lambda1, bearing_rad, d_rad):
    phi_C = np.arcsin(
        np.sin(phi1) * np.cos(d_rad)
        + np.cos(phi1) * np.sin(d_rad) * np.cos(bearing_rad)
    )

    y_lon = np.sin(bearing_rad) * np.sin(d_rad) * np.cos(phi1)
    x_lon = np.cos(d_rad) - np.sin(phi1) * np.sin(phi_C)
    d_lambda_AC = np.arctan2(y_lon, x_lon)

    lambda_C = lambda1 + d_lambda_AC

    return phi_C, lambda_C


def calculate_rhumb_line_track(phi1, lambda1, phi2, lambda2):
    """Calcule la Route de la Loxodromie (Track constante) en radians."""
    d_lambda = lambda2 - lambda1

    # Calcul de Delta M (différence de latitude méridienne)
    # Delta M = ln[tan(phi2/2 + pi/4) / tan(phi1/2 + pi/4)]
    epsilon = 1e-12

    tan_phi2 = np.tan(phi2 / 2 + np.pi / 4)
    tan_phi1 = np.tan(phi1 / 2 + np.pi / 4)

    d_m = np.log(np.maximum(tan_phi2, epsilon) / np.maximum(tan_phi1, epsilon))

    # Gestion des cas spéciaux (vol plein Est/Ouest)
    if np.abs(d_m) < epsilon:
        if np.abs(d_lambda) < epsilon:
            return np.nan
        return np.where(d_lambda > 0, np.pi / 2, -np.pi / 2)

    return np.arctan2(d_lambda, d_m)


# --- Scénario de Test et Tracé ---

# Points A et B (Exemple: New York -> Paris)
lat_A, lon_A = 40.7128, -74.0060  # New York (NYC)
lat_B, lon_B = 48.8566, 2.3522  # Paris (CDG)

# Conversion en radians
phi_A, lambda_A = to_radians(lat_A), to_radians(lon_A)
phi_B, lambda_B = to_radians(lat_B), to_radians(lon_B)

# 1. Calcul de la distance totale et de la Route Initiale
total_dist_km = great_circle_distance_km(phi_A, lambda_A, phi_B, lambda_B)
initial_gc_bearing_rad = calculate_initial_bearing(phi_A, lambda_A, phi_B, lambda_B)

# 2. Génération de N points intermédiaires (N=100)
N = 100
distances_traveled_km = np.linspace(0, total_dist_km, N)
distances_traveled_rad = distances_traveled_km / EARTH_RADIUS_KM

# 3. Calcul des coordonnées des N points intermédiaires (Orthodromie)
phi_C, lambda_C = calculate_intermediate_point(
    phi_A, lambda_A, initial_gc_bearing_rad, distances_traveled_rad
)

# 4. Calcul de la Route Courante de l'Orthodromie (Track Target dynamique C -> B)
gc_track_rad = calculate_current_bearing(phi_C, lambda_C, phi_B, lambda_B)
gc_track_deg = (to_degrees(gc_track_rad) + 360) % 360

# 5. Calcul de la Route Constante de la Loxodromie (Rhumb Line)
rl_track_rad = calculate_rhumb_line_track(phi_A, lambda_A, phi_B, lambda_B)
rl_track_deg = (to_degrees(rl_track_rad) + 360) % 360
rl_track_constant = np.full(N, rl_track_deg)

# --- Affichage des Résultats avec Matplotlib ---

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle(
    f"Comparaison de la Route (Track) : Orthodromie vs. Loxodromie\n(NYC à Paris - Distance: {total_dist_km:.0f} km)",
    fontsize=14,
)

# Graphique 1: Track (Route) en fonction de la Distance
ax1.plot(
    distances_traveled_km,
    gc_track_deg,
    label="Orthodromie (Route Cible Dynamique)",
    color="blue",
)
ax1.plot(
    distances_traveled_km,
    rl_track_constant,
    label="Loxodromie (Route Constante)",
    color="red",
    linestyle="--",
)
ax1.set_title("Route (Track) en fonction de la Distance Parcourue")
ax1.set_xlabel("Distance Parcourue (km)")
ax1.set_ylabel("Route (Track) en degrés (°)")
ax1.legend()
ax1.grid(True, linestyle="--")
ax1.set_ylim(min(gc_track_deg) * 0.9, max(gc_track_deg) * 1.1)

# Graphique 2: Latitude vs. Longitude (Visualisation de la Trajectoire)
ax2.plot(to_degrees(lambda_C), to_degrees(phi_C), label="Orthodromie", color="blue")
ax2.scatter(lon_A, lat_A, color="green", marker="o", label="Départ (A)", zorder=5)
ax2.scatter(lon_B, lat_B, color="red", marker="x", label="Arrivée (B)", zorder=5)
ax2.set_title("Trajectoire (Lat vs Lon)")
ax2.set_xlabel("Longitude (°)")
ax2.set_ylabel("Latitude (°)")
ax2.legend()
ax2.grid(True, linestyle="--")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# %%


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# --- Algorithme RDP (Ramer-Douglas-Peucker) ---


def perpendicular_distance(point, line_start, line_end):
    """
    Calcule la distance perpendiculaire d'un point à un segment de ligne.

    Args:
        point (np.ndarray): Coordonnées (x, y) du point.
        line_start (np.ndarray): Coordonnées (x, y) du début du segment.
        line_end (np.ndarray): Coordonnées (x, y) de la fin du segment.

    Returns:
        float: Distance perpendiculaire.
    """
    if np.array_equal(line_start, line_end):
        return euclidean(point, line_start)

    # Calcul de la distance d'un point à une ligne via les projections vectorielles
    line_vec = line_end - line_start
    point_vec = point - line_start

    line_len_sq = np.dot(line_vec, line_vec)
    t = np.dot(point_vec, line_vec) / line_len_sq

    # Gestion des projections en dehors du segment (on prend la distance au point le plus proche)
    if t < 0.0:
        return euclidean(point, line_start)
    elif t > 1.0:
        return euclidean(point, line_end)

    projection = line_start + t * line_vec
    return euclidean(point, projection)


def rdp(points, epsilon):
    """
    Simplifie une courbe en utilisant l'algorithme Ramer-Douglas-Peucker.

    Args:
        points (np.ndarray): Tableau des points de la trajectoire (N, 2).
        epsilon (float): Seuil de tolérance (distance maximale autorisée).

    Returns:
        list: Indices des points conservés (les points tournants/pivots).
    """

    if len(points) <= 2:
        return list(range(len(points)))

    # Trouver le point ayant la distance maximale
    d_max = 0.0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > d_max:
            index = i
            d_max = d

    # Si la distance maximale est supérieure à epsilon, ce point est un pivot
    if d_max > epsilon:
        # Simplifier récursivement les segments avant et après le pivot
        results1 = rdp(points[: index + 1], epsilon)
        results2 = rdp(points[index:], epsilon)

        # Combiner les résultats (en évitant le double comptage du pivot 'index')
        # On décale les indices de results2 et retire le premier (qui est 'index')
        results = results1[:-1] + [i + index for i in results2]
        return results
    else:
        # Si d_max <= epsilon, le segment peut être approximé par une ligne droite
        return [0, end]


# --- Fonction de Traitement ADS-B (pour l'exemple) ---


def identify_turning_points_adsb(latitudes, longitudes, epsilon_km=0.5):
    """
    Prépare les données de trajectoire et appelle l'algorithme RDP.

    Args:
        latitudes (np.ndarray): Tableau des latitudes.
        longitudes (np.ndarray): Tableau des longitudes.
        epsilon_km (float): Seuil de tolérance en kilomètres.

    Returns:
        np.ndarray: Indices des points tournants.
    """

    # 1. Conversion Lat/Lon en coordonnées planes (approximation locale en km)
    # Utilisez le premier point comme origine (0, 0)
    R_earth = 6371.0  # Rayon terrestre moyen en km

    # Déplacement Latéral (Nord-Sud) - 1 deg de Lat est approx 111 km
    y_coords_km = (latitudes - latitudes[0]) * 111.0

    # Déplacement Longitudinal (Est-Ouest) - dépend du cos(latitude)
    lat_rad = np.radians(latitudes)
    x_coords_km = (longitudes - longitudes[0]) * (
        R_earth * np.cos(lat_rad) * np.pi / 180.0
    )

    # Combinaison des coordonnées
    points = np.stack((x_coords_km, y_coords_km), axis=1)

    # 2. Application de l'algorithme RDP
    pivot_indices = rdp(points, epsilon_km)

    return np.array(pivot_indices)


latitudes = res.latitude.values
longitudes = res.longitude.values

# 2. Détection des points tournants (seuil de 1.0 km)
epsilon_km = 0.1  # Le seuil de tolérance doit être ajusté pour votre type de vol
pivot_indices = identify_turning_points_adsb(
    latitudes, longitudes, epsilon_km=epsilon_km
)

print(f"Nombre total de points : {len(latitudes)}")
print(f"Seuil RDP (Epsilon) : {epsilon_km} km")
print(f"Indices des points tournants/pivots identifiés : {pivot_indices}")
print(f"Nombre de segments clés : {len(pivot_indices) - 1}")


# 3. Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.plot(
    longitudes,
    latitudes,
    "k.",
    label="Trajectoire Originale (ADS-B + Bruit)",
    alpha=0.5,
)

# Tracé de la ligne simplifiée (segments clés)
simplified_lons = longitudes[pivot_indices]
simplified_lats = latitudes[pivot_indices]
plt.plot(
    simplified_lons,
    simplified_lats,
    "r-",
    linewidth=2,
    label="Trajectoire Simplifiée (RDP)",
)

# Mise en évidence des points tournants
plt.plot(
    simplified_lons,
    simplified_lats,
    "bo",
    markersize=8,
    label="Points Tournants (Pivots)",
)

plt.title(
    f"Identification des Points Tournants par Ramer-Douglas-Peucker (Epsilon={epsilon_km} km)"
)
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.legend()
plt.grid(True)
plt.show()
# %%
import numpy as np
from scipy.signal import savgol_filter, find_peaks


def detect_centered_turning_points(
    track_degrees, time_seconds, threshold_deg_per_sec=0.1
):
    """
    Détecte les points tournants (centrés) en utilisant la vitesse de rotation.

    Args:
        track_degrees (np.ndarray): Tableau de la Route Sol (Track) en degrés.
        time_seconds (np.ndarray): Tableau des timestamps en secondes.
        threshold_deg_per_sec (float): Seuil minimum de rotation pour considérer un virage.

    Returns:
        np.ndarray: Indices des points centraux de chaque virage.
    """

    # 1. Lissage de la Track pour réduire le bruit (Filtre Savitzky-Golay)
    # window_length doit être impair. polyorder doit être inférieur à window_length.
    # Adapter la fenêtre à la fréquence de vos données ADS-B (ex: 5 points)
    window_length = 9
    polyorder = 3
    if len(track_degrees) < window_length:
        print("Erreur: Données trop courtes pour le lissage.")
        return np.array([])

    smoothed_track = savgol_filter(track_degrees, window_length, polyorder)

    # 2. Calcul du Taux de Rotation (Dérivée temporelle de la Track)
    # Utilisation de np.gradient pour calculer la dérivée vectorisée
    # Correction de l'angle (pour gérer les sauts de 360 à 0 degrés)
    d_track = np.diff(smoothed_track, prepend=smoothed_track[0])
    d_track = np.where(d_track > 180, d_track - 360, d_track)
    d_track = np.where(d_track < -180, d_track + 360, d_track)

    # Calcul de Delta T (différence de temps)
    dt = np.diff(time_seconds, prepend=1)

    # Taux de rotation en degrés/seconde
    rotation_rate = d_track / dt

    # 3. Détection des Pics (Centres des virages)

    # Utiliser la valeur absolue car le pic peut être positif (droite) ou négatif (gauche)
    abs_rotation_rate = np.abs(rotation_rate)

    # Détection des sommets au-dessus du seuil
    # height: Hauteur minimale du pic
    # distance: Distance minimale entre deux pics (pour éviter de détecter le même virage deux fois)
    peaks, properties = find_peaks(
        abs_rotation_rate, height=threshold_deg_per_sec, distance=10
    )  # 10 points de distance minimum entre les virages

    return peaks


track_simulated = np.unwrap(res["track"].values)
time_data = (res["timestamp"] - res["timestamp"].iloc[0]).dt.total_seconds().values
# Détection des points centraux (utiliser un seuil bas car les données sont lissées)
# Le seuil doit être adapté à la manœuvre typique de l'avion (ex: 0.2 deg/sec)
turning_indices = detect_centered_turning_points(
    track_simulated, time_data, threshold_deg_per_sec=0.1
)

# --- Affichage des résultats ---
plt.figure(figsize=(10, 6))
# Calculer le taux de rotation pour l'affichage (réutilise la logique interne)
smoothed_track = savgol_filter(track_simulated, 9, 3)
d_track = np.diff(smoothed_track, prepend=smoothed_track[0])
dt = np.diff(time_data, prepend=1)
rotation_rate = np.abs(d_track / dt)

plt.plot(
    time_data, rotation_rate, "g-", alpha=0.7, label="|Taux de Rotation| (deg/sec)"
)
plt.scatter(
    time_data[turning_indices],
    rotation_rate[turning_indices],
    color="red",
    s=80,
    marker="o",
    label="Points Centraux du Virage",
)
plt.axhline(0.2, color="gray", linestyle="--", label="Seuil de Détection")
plt.title("Détection des Points Tournants Centrés par Vitesse de Rotation")
plt.xlabel("Temps (secondes)")
plt.ylabel("Vitesse de Rotation Absolue |d(Track)/dt|")
plt.legend()
plt.grid(True)
plt.show()

print(f"Indices des points centraux (pics de rotation) : {turning_indices}")
# %%

import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import pandas as pd


def detect_centered_turning_points(
    track_degrees, time_seconds, threshold_deg_per_sec=0.1
):
    """
    Détecte les points tournants (centrés) en utilisant la vitesse de rotation.
    [... (la définition de la fonction est omise ici pour la concision, car elle est inchangée)]
    """
    window_length = 15
    polyorder = 3
    if len(track_degrees) < window_length:
        print("Erreur: Données trop courtes pour le lissage.")
        return np.array([])

    # 1. Lissage de la Track
    smoothed_track = savgol_filter(track_degrees, window_length, polyorder)

    # 2. Calcul du Taux de Rotation (Dérivée temporelle de la Track)
    d_track = np.diff(smoothed_track, prepend=smoothed_track[0])
    # Correction de l'angle 360/0
    d_track = np.where(d_track > 180, d_track - 360, d_track)
    d_track = np.where(d_track < -180, d_track + 360, d_track)

    dt = np.diff(time_seconds, prepend=1)
    rotation_rate = d_track / dt

    # 3. Détection des Pics (Centres des virages)
    abs_rotation_rate = np.abs(rotation_rate)

    peaks, properties = find_peaks(
        abs_rotation_rate, height=threshold_deg_per_sec, distance=10
    )  # 10 points de distance minimum entre les virages

    # Retourne les indices ET la Track LISSEE (pour l'affichage)
    return peaks, smoothed_track


i = 0
if True:
    # for f_id, f in tasks:
    print(f_id)
    res = build_spd_and_vert_selected_from_segments(f, selected_param_config)

    # --- Application de la Détection ---
    track_simulated = np.unwrap(res["track"].values)
    time_data = (res["timestamp"] - res["timestamp"].iloc[0]).dt.total_seconds().values

    # Détection des points centraux
    threshold = 0.05
    turning_indices, smoothed_track = detect_centered_turning_points(
        track_simulated, time_data, threshold_deg_per_sec=threshold
    )

    # --- Affichage des résultats sur la Courbe de Track ---
    plt.figure(figsize=(10, 6))

    # 1. Affichage de la Track Lissée
    plt.plot(
        time_data, smoothed_track, "b-", label="Route Sol Lissée (Track)", linewidth=2
    )

    # 2. Mise en évidence des Points Tournants Centrés
    plt.scatter(
        time_data[turning_indices],
        smoothed_track[turning_indices],
        color="red",
        s=100,
        marker="o",
        label="Points Centraux du Virage (Cibles Hdg)",
    )

    plt.scatter(
        time_data[turning_indices[6:8]],
        smoothed_track[turning_indices[6:8]],
        color="g",
        s=100,
        marker="o",
        label="Points Centraux du Virage (Cibles Hdg)",
    )

    # 3. Affichage du Bruit (la Track simulée brute)
    plt.plot(
        time_data, track_simulated, "k.", alpha=0.3, label="Track Brute (avec Bruit)"
    )

    plt.title(
        f"Identification des Cibles de Pilotage sur la Courbe de Route (Track)\nSeuil de Vitesse de Rotation: {threshold}°/sec"
    )
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Route Sol (Track) en degrés (°)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.plot(res["longitude"], res["latitude"], "k", alpha=0.5, lw=1)
    plt.scatter(
        res["longitude"].values[turning_indices],
        res["latitude"].values[turning_indices],
        color="red",
        s=10,
        marker="o",
    )

    i += 1
    # if i >= 4:
    #    break

# %%

lon_A, lon_B = res["longitude"].values[turning_indices[6:8]]
lat_A, lat_B = res["latitude"].values[turning_indices[6:8]]
# Conversion en radians
phi_A, lambda_A = to_radians(lat_A), to_radians(lon_A)
phi_B, lambda_B = to_radians(lat_B), to_radians(lon_B)

# 1. Calcul de la distance totale et de la Route Initiale
total_dist_km = great_circle_distance_km(phi_A, lambda_A, phi_B, lambda_B)
initial_gc_bearing_rad = calculate_initial_bearing(phi_A, lambda_A, phi_B, lambda_B)

# 2. Génération de N points intermédiaires (N=100)
N = 100
distances_traveled_km = np.linspace(0, total_dist_km, N)
distances_traveled_rad = distances_traveled_km / EARTH_RADIUS_KM

# 3. Calcul des coordonnées des N points intermédiaires (Orthodromie)
phi_C, lambda_C = calculate_intermediate_point(
    phi_A, lambda_A, initial_gc_bearing_rad, distances_traveled_rad
)

# 4. Calcul de la Route Courante de l'Orthodromie (Track Target dynamique C -> B)
gc_track_rad = calculate_current_bearing(phi_C, lambda_C, phi_B, lambda_B)
gc_track_deg = (to_degrees(gc_track_rad) + 360) % 360

# 5. Calcul de la Route Constante de la Loxodromie (Rhumb Line)
rl_track_rad = calculate_rhumb_line_track(phi_A, lambda_A, phi_B, lambda_B)
rl_track_deg = (to_degrees(rl_track_rad) + 360) % 360
rl_track_constant = np.full(N, rl_track_deg)

# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# R_earth = 6371.0 # Déjà défini dans les fonctions (assumons qu'elles sont dans le scope)


def calculate_successive_distances(latitudes, longitudes):
    """Calcule la distance Great-Circle entre chaque point successif."""
    phi1 = np.radians(latitudes[:-1])
    lambda1 = np.radians(longitudes[:-1])
    phi2 = np.radians(latitudes[1:])
    lambda2 = np.radians(longitudes[1:])

    # Loi des Cosinus Sphérique (vectorisée)
    d_sigma = np.arccos(
        np.sin(phi1) * np.sin(phi2)
        + np.cos(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)
    )
    # Remplacer les valeurs NaN/très proches de zéro par une petite distance
    d_sigma = np.nan_to_num(d_sigma, nan=0.0)

    return d_sigma * 6371.0  # Assumer R=6371km


def get_cumulative_distance(df, start_index, end_index):
    """Calcule la distance cumulée pour un segment de trajectoire."""
    lat_segment = df["latitude"].values[start_index : end_index + 1]
    lon_segment = df["longitude"].values[start_index : end_index + 1]

    distances_successives = calculate_successive_distances(lat_segment, lon_segment)

    # Ajouter 0 au début pour le point de départ
    cumulative_distances = np.insert(np.cumsum(distances_successives), 0, 0)

    return cumulative_distances


# Assurez-vous que les fonctions to_radians, great_circle_distance_km,
# calculate_initial_bearing, calculate_intermediate_point, calculate_current_bearing,
# et calculate_rhumb_line_track sont définies dans votre environnement.

# --- Définition des Points du Segment (Utilisation des indices 6 à 8) ---
for i in range(1, 2):
    start_idx = turning_indices[i]
    end_idx = turning_indices[1 + i]

    lon_A = res["longitude"].values[start_idx]
    lat_A = res["latitude"].values[start_idx]
    lon_B = res["longitude"].values[end_idx] + 3.25
    lat_B = res["latitude"].values[end_idx]

    # --- Calcul de la Trajectoire Idéale (Orthodromie) ---
    phi_A, lambda_A = to_radians(lat_A), to_radians(lon_A)
    phi_B, lambda_B = to_radians(lat_B), to_radians(lon_B)

    # 1. Distance totale et Route Initiale
    total_dist_km = great_circle_distance_km(phi_A, lambda_A, phi_B, lambda_B)
    initial_gc_bearing_rad = calculate_initial_bearing(phi_A, lambda_A, phi_B, lambda_B)

    # 2. Génération de N points intermédiaires (N=100) pour l'Orthodromie
    N_ideal = 100
    distances_traveled_km_ideal = np.linspace(0, total_dist_km, N_ideal)
    distances_traveled_rad_ideal = distances_traveled_km_ideal / EARTH_RADIUS_KM

    # 3. Coordonnées et Route de l'Orthodromie Calculée
    phi_C, lambda_C = calculate_intermediate_point(
        phi_A, lambda_A, initial_gc_bearing_rad, distances_traveled_rad_ideal
    )
    gc_track_rad = calculate_current_bearing(phi_C, lambda_C, phi_B, lambda_B)
    gc_track_deg = (to_degrees(gc_track_rad) + 360) % 360

    # --- Calcul de la Trajectoire Réelle Volée ---

    # 1. Extraction des données réelles du segment
    actual_track_segment = res["track"].values[start_idx : end_idx + 1]
    actual_lat_segment = res["latitude"].values[start_idx : end_idx + 1]
    actual_lon_segment = res["longitude"].values[start_idx : end_idx + 1]

    # 2. Calcul de la Distance Cumulée (Axe X pour la courbe réelle)
    distances_km_real = get_cumulative_distance(res, start_idx, end_idx)

    # --- Tracé de Comparaison ---
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    ax1.plot(
        distances_traveled_km_ideal,
        np.unwrap(gc_track_deg),
        label="Orthodromie Calculée (Chemin Idéal)",
        color="blue",
        linewidth=2,
    )
    if False:
        ax1.plot(
            distances_km_real,
            actual_track_segment,
            label="Route Volée Réelle (ADS-B)",
            color="red",
            linestyle="--",
            alpha=0.7,
        )

    ax1.set_title(
        f"Comparaison de la Route Volée vs. Route Cible Idéale (Distance: {total_dist_km:.1f} km)"
    )
    ax1.set_xlabel("Distance Parcourue Cumulée (km)")
    ax1.set_ylabel("Route (Track) en degrés (°)")
    ax1.legend()
    ax1.grid(True, linestyle="--")

    plt.show()
# %%


# %%
