import pandas as pd
import numpy as np
import math
import random

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Ruta al archivo de accidentes (ajústala a tu compu)
DATA_CSV = "accidentes_Ags_2023.csv"

# Radio de cobertura en km (para la competencia: 2 km)
RADIO_KM = 2.0

# Paso de la malla en km (aquí = radio/2 → 1 km)
GRID_STEP_KM = RADIO_KM / 2.0

# Número de ambulancias para las que generaremos soluciones
N_AMBULANCIAS_LIST = [3, 5, 7, 10]

# Parámetros del GRASP
ALPHA = 0.3       # 0 = súper greedy, 1 = muy random
NUM_ITER = 40     # iteraciones de GRASP (multiarranque)

SEED = 42         # para reproducibilidad
random.seed(SEED)
np.random.seed(SEED)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def robust_read_csv(path):
    """Intenta leer el CSV con varias codificaciones/separadores."""
    for enc in ["latin-1", "cp1252", "utf-8"]:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                return df
            except Exception:
                continue
    # Último intento genérico
    return pd.read_csv(path, engine="python")

def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km entre arrays lat1/lon1 y lat2/lon2."""
    R = 6371.0  # radio de la Tierra en km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

# ==========================================
# CARGA DE DATOS
# ==========================================
print("Cargando datos de accidentes...")
df = robust_read_csv(DATA_CSV)
df = df.dropna(subset=["LATITUD", "LONGITUD"]).copy()

acc_lats = df["LATITUD"].to_numpy()
acc_lons = df["LONGITUD"].to_numpy()
N_ACC = len(df)
print(f"Accidentes cargados: {N_ACC}")

# ==========================================
# CONSTRUCCIÓN DE LA MALLA DE CANDIDATOS
# ==========================================
print("\nConstruyendo malla de candidatos...")

lat_min, lat_max = df["LATITUD"].min(), df["LATITUD"].max()
lon_min, lon_max = df["LONGITUD"].min(), df["LONGITUD"].max()

# Conversión aproximada de km a grados
lat0 = df["LATITUD"].median()
km_per_deg_lat = 111.0
km_per_deg_lon = 111.320 * math.cos(math.radians(lat0))

dlat_step = GRID_STEP_KM / km_per_deg_lat
dlon_step = GRID_STEP_KM / km_per_deg_lon

i_vals = np.arange(lat_min, lat_max + dlat_step, dlat_step)
j_vals = np.arange(lon_min, lon_max + dlon_step, dlon_step)

candidatos = [(lat_c, lon_c) for lat_c in i_vals for lon_c in j_vals]

print(f"  Candidatos generados: {len(candidatos)}")

# ==========================================
# COBERTURAS POR CANDIDATO (R = 2 km)
# ==========================================
print("\nCalculando cobertura de cada candidato (R = 2 km)...")

coberturas_candidatos = []
for (lat_c, lon_c) in candidatos:
    dists = haversine_km(lat_c, lon_c, acc_lats, acc_lons)
    cubiertos = np.where(dists <= RADIO_KM)[0]
    if len(cubiertos) > 0:
        coberturas_candidatos.append({
            "lat": lat_c,
            "lon": lon_c,
            "cubiertos": frozenset(cubiertos),
            "score": len(cubiertos)
        })

print(f"  Candidatos útiles (cobertura > 0): {len(coberturas_candidatos)}")
print(f"  Máximo accidentes cubiertos por un solo candidato: {max(c['score'] for c in coberturas_candidatos)}")

# ==========================================
# GRASP + BÚSQUEDA LOCAL
# ==========================================
def construir_solucion_grasp(max_amb, alpha=0.3):
    """
    Construcción greedy-aleatoria tipo GRASP:
    - max_amb: número de ambulancias
    - alpha: controla tamaño de la RCL
    """
    cubiertos_global = set()
    solucion = []
    indices_disponibles = list(range(len(coberturas_candidatos)))

    for k in range(max_amb):
        ganancias = []
        for idx in indices_disponibles:
            nuevos = coberturas_candidatos[idx]["cubiertos"] - cubiertos_global
            ganancias.append((idx, len(nuevos)))

        # Quitar candidatos que ya no aportan nada
        ganancias = [g for g in ganancias if g[1] > 0]
        if not ganancias:
            break

        # Ordenar por ganancia
        ganancias.sort(key=lambda x: x[1], reverse=True)
        max_gain = ganancias[0][1]
        min_gain = ganancias[-1][1]

        # RCL: aquellos con ganancia >= umbral
        umbral = max_gain - alpha * (max_gain - min_gain)
        rcl = [(idx, g) for idx, g in ganancias if g >= umbral]

        # Elegimos uno aleatorio de la RCL
        idx_elegido, _ = random.choice(rcl)
        cand = coberturas_candidatos[idx_elegido]

        solucion.append(cand)
        cubiertos_global |= cand["cubiertos"]
        indices_disponibles.remove(idx_elegido)

    return solucion, len(cubiertos_global), cubiertos_global

def busqueda_local_simple(solucion, cubiertos_global, max_iter=30):
    """
    Búsqueda local sencilla:
    - Quita una ambulancia y prueba reemplazarla por otro candidato
      que mejore la cobertura total.
    """
    mejor_sol = solucion[:]
    mejor_cubiertos = set(cubiertos_global)
    mejor_score = len(mejor_cubiertos)

    for _ in range(max_iter):
        if not mejor_sol:
            break

        # Elegimos una ambulancia para sacar
        idx_out = random.randrange(len(mejor_sol))

        # Cobertura sin esa ambulancia
        cubiertos_sin = set()
        for i, c in enumerate(mejor_sol):
            if i == idx_out:
                continue
            cubiertos_sin |= c["cubiertos"]

        # Buscamos un candidato que aporte la mayor mejora
        mejor_mejora = 0
        mejor_in = None
        for cand in coberturas_candidatos:
            if cand in mejor_sol:
                continue
            nuevos = cand["cubiertos"] - cubiertos_sin
            mejora = len(nuevos)
            if mejora > mejor_mejora:
                mejor_mejora = mejora
                mejor_in = cand

        # Si encontramos mejora, aceptamos movimiento
        if mejor_in is not None and len(cubiertos_sin | mejor_in["cubiertos"]) > mejor_score:
            mejor_sol[idx_out] = mejor_in
            mejor_cubiertos = cubiertos_sin | mejor_in["cubiertos"]
            mejor_score = len(mejor_cubiertos)

    return mejor_sol, mejor_score, mejor_cubiertos

def resolver_mcp_grasp(n_ambulancias, num_iter=40, alpha=0.3):
    """
    Multi-arranque GRASP + búsqueda local.
    """
    mejor_sol_global = None
    mejor_score_global = -1
    mejor_cubiertos_global = set()

    for it in range(num_iter):
        sol, score, cubiertos = construir_solucion_grasp(n_ambulancias, alpha=alpha)
        sol_ls, score_ls, cubiertos_ls = busqueda_local_simple(sol, cubiertos, max_iter=20)

        if score_ls > mejor_score_global:
            mejor_score_global = score_ls
            mejor_sol_global = sol_ls
            mejor_cubiertos_global = cubiertos_ls

    return mejor_sol_global, mejor_score_global, mejor_cubiertos_global

# ==========================================
# RESOLVER PARA VARIOS N DE AMBULANCIAS
# ==========================================
print("\nResolviendo MCP con GRASP + búsqueda local (R = 2 km)...")

for n_amb in N_AMBULANCIAS_LIST:
    print(f"\n===== N_AMBULANCIAS = {n_amb} =====")
    sol, score, cubiertos = resolver_mcp_grasp(n_amb, num_iter=NUM_ITER, alpha=ALPHA)

    porcentaje = score / N_ACC * 100.0
    print(f"Cobertura: {score} de {N_ACC} accidentes ({porcentaje:.2f}%)")

    # Construimos DataFrame de salida en formato MCP
    ambulancias = []
    for c in sol:
        ambulancias.append({
            "LATITUD": c["lat"],
            "LONGITUD": c["lon"]
        })
    df_sol = pd.DataFrame(ambulancias)

    # Guardar CSV para subir a la página
    nombre_archivo = f"solucion_mcp_{n_amb}ambulancias_2km.csv"
    df_sol.to_csv(nombre_archivo, index=False)
    print(f"Archivo guardado: {nombre_archivo}")

print("\n¡Listo! Ya puedes subir los CSV a la página de la competencia.")