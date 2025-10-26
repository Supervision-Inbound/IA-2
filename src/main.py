# (Imports y configuración global - v7 compatible)
import os
import argparse
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# --- MIOS (Para replicar inferencia_core) ---
from src.inferencia.features import ensure_ts, add_time_parts, add_lags_mas, dummies_and_reindex # Usaremos estas helpers
from src.inferencia.erlang import required_agents, schedule_agents # Para agentes
# --- FIN MIOS ---

# --- CONFIGURACIÓN GLOBAL ---
TZ = 'America/Santiago'; os.environ['TZ'] = TZ
BASE_DIR = os.path.dirname(os.path.abspath(__file__)); ROOT_DIR = os.path.dirname(BASE_DIR)

# === INICIO CAMBIO: Ajustar rutas a la estructura src/ ===
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public") # Antes estaba en ROOT_DIR, ahora consistente
# === FIN CAMBIO ===

# Archivos (nombres estándar)
PLANNER_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_planner.keras"); PLANNER_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_planner.pkl"); PLANNER_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_planner.json")
RISK_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_riesgos.keras"); RISK_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_riesgos.pkl"); RISK_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_riesgos.json"); RISK_BASELINES_FILE = os.path.join(MODEL_DIR, "baselines_clima.pkl")
TMO_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_tmo.keras"); TMO_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_tmo.pkl"); TMO_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_tmo.json")

HOSTING_FILE = os.path.join(DATA_DIR, "historical_data.csv")
# === INICIO CAMBIO: Usar el nombre de archivo correcto para TMO ===
# Asegúrate que este nombre coincida EXACTAMENTE con tu archivo
TMO_FILE = os.path.join(DATA_DIR, "TMO_HISTORICO.csv") # O "HISTORICO_TMO.csv"
# === FIN CAMBIO ===
FERIADOS_FILE = os.path.join(DATA_DIR, "Feriados_Chilev2.csv")
CLIMA_HIST_FILE = os.path.join(DATA_DIR, "historical_data.csv") # Para simular clima

TARGET_CALLS = "recibidos_nacional"; TARGET_TMO = "tmo_general"

# --- INICIO CAMBIO: Añadir variables de inferencia_core ---
HIST_WINDOW_DAYS = 90
DIAS_DE_PAGO = {1, 2, 15, 16, 29, 30, 31}
# --- FIN CAMBIO ---

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'; warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl'); warnings.filterwarnings('ignore', category=FutureWarning)


# --- FUNCIONES DE UTILIDAD ---

# (read_data - sin cambios)
def read_data(path, hoja=None):
    path_lower = path.lower();
    if not os.path.exists(path): raise FileNotFoundError(f"No encontrado: {path}.")
    if path_lower.endswith(".csv"):
        try: df = pd.read_csv(path, low_memory=False);
        except Exception: df = None
        if df is None or (df.shape[1] == 1 and df.iloc[0,0] is not None and ';' in str(df.iloc[0,0])):
             try: df = pd.read_csv(path, delimiter=';', low_memory=False)
             except Exception as e2: raise ValueError(f"No se pudo leer {path}: {e2}")
        return df
    elif path_lower.endswith((".xlsx", ".xls")): return pd.read_excel(path, sheet_name=hoja if hoja is not None else 0)
    else: raise ValueError(f"Formato no soportado: {path}")

# === INICIO CAMBIO: Usar ensure_ts de features.py (más robusto) ===
# def ensure_ts_and_tz(df):
#     # ... (código original eliminado)
#     pass # Usaremos la versión importada
# === FIN CAMBIO ===

# --- INICIO CAMBIO: Usar add_time_parts de features.py ---
# def add_time_parts(df):
#    # ... (código original eliminado)
#    pass # Usaremos la versión importada
# --- FIN CAMBIO ---

# (normalize_climate_columns - sin cambios)
def normalize_climate_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {'temperatura': ['temperature_2m', 'temperatura', 'temp', 'temp_2m'], 'precipitacion': ['precipitation', 'precipitacion', 'precipitación', 'rain_mm', 'rain'], 'lluvia': ['rain', 'lluvia', 'rainfall']}
    df_renamed = df.copy(); df_renamed.columns = [c.lower().strip().replace(' ', '_') for c in df_renamed.columns]
    for std, poss in column_map.items():
        for name in poss:
            if name in df_renamed.columns: df_renamed.rename(columns={name: std}, inplace=True); break
    return df_renamed

# --- INICIO CAMBIO: Usar calculate_erlang_agents de erlang.py ---
# def calculate_erlang_agents(calls_per_hour, tmo_seconds, occupancy_target=0.85):
#     # ... (código original eliminado)
#     pass # Usaremos la versión importada
# --- FIN CAMBIO ---

# --- INICIO CAMBIO: Añadir helpers de inferencia_core ---
def _load_cols(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _is_holiday(ts, holidays_set: set) -> int:
    if not holidays_set:
        return 0
    try:
        # Intentar convertir a TZ local primero
        d = ts.tz_convert(TZ).date()
    except Exception:
        # Si falla (ej. ya es date o no tiene tz), usar date directamente
        d = ts.date()
    return 1 if d in holidays_set else 0

def _is_payday(ts) -> int:
    return 1 if ts.day in DIAS_DE_PAGO else 0
# --- FIN CAMBIO ---


# --- FUNCIONES DEL PIPELINE DE INFERENCIA ---

# (fetch_future_weather, process_future_climate, generate_alerts_json - sin cambios)
def fetch_future_weather(start_date, end_date):
    print("     [Clima] SIMULANDO API de clima futuro...");
    try: df_hist = read_data(CLIMA_HIST_FILE)
    except FileNotFoundError: print(f"     [Clima] ADVERTENCIA: {CLIMA_HIST_FILE} no encontrado. Dummy."); comunas = ['Santiago']; dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts'])); df_simulado['temperatura'] = 15; df_simulado['precipitacion'] = 0; df_simulado['lluvia'] = 0; return df_simulado.reset_index()

    # --- INICIO CAMBIO: Usar ensure_ts ---
    try:
        df_hist = ensure_ts(df_hist) # ensure_ts maneja TZ
    except ValueError as e:
        print(f"     [Clima] ADVERTENCIA: Error procesando ts en {CLIMA_HIST_FILE}: {e}. Dummy.")
        # ... (código dummy igual que antes) ...
        comunas = ['Santiago']; dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts'])); df_simulado['temperatura'] = 15; df_simulado['precipitacion'] = 0; df_simulado['lluvia'] = 0; return df_simulado.reset_index()
    # --- FIN CAMBIO ---

    df_hist = normalize_climate_columns(df_hist); climate_cols_found = [col for col in ['temperatura', 'precipitacion', 'lluvia'] if col in df_hist.columns]

    if not climate_cols_found: print(f"     [Clima] ADVERTENCIA: No cols clima en {CLIMA_HIST_FILE}. Dummy."); comunas = ['Santiago']; dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts'])); df_simulado['temperatura'] = 15; df_simulado['precipitacion'] = 0; df_simulado['lluvia'] = 0; return df_simulado.reset_index()

    if 'comuna' not in df_hist.columns: print("     [Clima] ADVERTENCIA: 'comuna' no encontrada. Dummy 'Santiago'."); df_hist['comuna'] = 'Santiago'

    future_dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_future_list = []

    # --- INICIO CAMBIO: Usar index para búsqueda más rápida ---
    df_hist_indexed = df_hist.set_index('ts')
    # --- FIN CAMBIO ---

    for date in future_dates:
        # --- INICIO CAMBIO: Búsqueda año anterior más robusta ---
        sim_date_y_minus_1 = date - pd.DateOffset(years=1)
        # Buscar la hora exacta o la más cercana si no existe
        try:
            # Usamos get_loc con method='nearest' para encontrar la más cercana
            idx_loc = df_hist_indexed.index.get_indexer([sim_date_y_minus_1], method='nearest')
            # Extraer el timestamp real encontrado
            sim_date_found = df_hist_indexed.index[idx_loc][0]
            # Verificar si está razonablemente cerca (ej. +/- 12 horas) para evitar saltos grandes
            if abs((sim_date_found - sim_date_y_minus_1).total_seconds()) <= 12 * 3600:
                 data_sim = df_hist_indexed.loc[[sim_date_found]].reset_index() # .loc requiere lista para mantener formato
            else:
                 data_sim = pd.DataFrame() # No se encontró fecha cercana
        except (KeyError, IndexError):
            data_sim = pd.DataFrame() # No se encontró la fecha
        # --- FIN CAMBIO ---

        if not data_sim.empty: data_sim = data_sim.copy(); data_sim['ts'] = date; df_future_list.append(data_sim)

    if not df_future_list:
        print("     [Clima] No match año anterior. Usando última semana.");
        last_ts_hist = df_hist_indexed.index.max()
        if last_ts_hist is None: # Si df_hist_indexed está vacío
             print("     [Clima] ADVERTENCIA: Histórico vacío. Dummy.")
             return fetch_future_weather(start_date, end_date) # Llamada recursiva para dummy

        last_week_start = last_ts_hist - pd.Timedelta(days=7)
        last_week = df_hist_indexed.loc[last_week_start:].reset_index() # reset_index para tener 'ts' como columna

        if last_week.empty: print("     [Clima] ADVERTENCIA: No datos última semana. Dummy."); return fetch_future_weather(start_date, end_date)

        # --- INICIO CAMBIO: Mapeo última semana más robusto ---
        last_week['day_hour'] = last_week['ts'].dt.dayofweek * 100 + last_week['ts'].dt.hour
        day_hour_map = last_week.set_index('day_hour')
        # --- FIN CAMBIO ---

        for date in future_dates:
             # --- INICIO CAMBIO: Usar mapeo day_hour ---
             current_day_hour = date.dayofweek * 100 + date.hour
             if current_day_hour in day_hour_map.index:
                 # Puede haber múltiples entradas para el mismo day_hour, tomamos la primera
                 data_sim = day_hour_map.loc[[current_day_hour]].iloc[[0]].reset_index(drop=True)
                 data_sim['ts'] = date
                 df_future_list.append(data_sim)
             # --- FIN CAMBIO ---

    if not df_future_list: print("     [Clima] ADVERTENCIA: Falló simulación. Dummy."); return fetch_future_weather(start_date, end_date) # Llamada recursiva

    df_simulado = pd.concat(df_future_list); all_comunas = df_hist['comuna'].unique(); all_dates = future_dates; full_index = pd.MultiIndex.from_product([all_comunas, all_dates], names=['comuna', 'ts'])

    df_final = df_simulado.set_index(['comuna', 'ts']).reindex(full_index); df_final = df_final.groupby(level='comuna').ffill().bfill(); df_final = df_final.fillna(0);

    print(f"     [Clima] Simulación API completada. {len(df_final)} registros."); return df_final.reset_index()


def process_future_climate(df_future_weather, df_baselines):
    print("     [Clima] Procesando datos futuros y calculando anomalías..."); df = normalize_climate_columns(df_future_weather.copy())

    # --- INICIO CAMBIO: Usar ensure_ts ---
    # (El bloque original de manejo de TZ se elimina, ensure_ts lo hace)
    df = ensure_ts(df)
    # --- FIN CAMBIO ---

    df = df.dropna(subset=['ts']).sort_values(['comuna', 'ts'])
    # --- INICIO CAMBIO: Usar add_time_parts ---
    df = add_time_parts(df) # Esto añade dow, hour, etc.
    # --- FIN CAMBIO ---

    if df_baselines.empty: print("     [Clima] Adv: Baselines vacíos, anomalías serán 0."); df_merged = df.copy(); df_merged['temperatura_median'] = 0; df_merged['temperatura_std'] = 1 # Etc
    # --- INICIO CAMBIO: Asegurar que columnas de baseline existan ---
    elif not all(c in df_baselines.columns for c in ['comuna_', 'dow_', 'hour_']):
         print("     [Clima] Adv: Baselines inválidos (faltan comuna_/dow_/hour_). Anomalías serán 0."); df_merged = df.copy()
    else:
        # Asegurar que las columnas para el merge existan en df
        if 'dow' not in df.columns or 'hour' not in df.columns:
            print("     [Clima] Adv: Faltan 'dow' u 'hour' en datos de clima. Anomalías serán 0.")
            df_merged = df.copy()
        else:
             # Renombrar columnas de baseline para evitar sufijos
             baseline_cols_map = {c: c.replace('_median', '').replace('_std', '').replace('_', '') for c in df_baselines.columns if c not in ['comuna_', 'dow_', 'hour_']}
             df_baselines_renamed = df_baselines.rename(columns={**baseline_cols_map, 'comuna_': 'comuna', 'dow_': 'dow', 'hour_': 'hour'})

             df_merged = pd.merge(df, df_baselines_renamed, on=['comuna', 'dow', 'hour'], how='left')
    # --- FIN CAMBIO ---

    numeric_cols = df_merged.select_dtypes(include=np.number).columns; df_merged[numeric_cols] = df_merged[numeric_cols].fillna(df_merged[numeric_cols].mean())

    expected_metrics = [c for c in ['temperatura', 'precipitacion', 'lluvia'] if c in df.columns]; anomaly_cols = []

    for metric in expected_metrics:
        median_col = f'{metric}median'; std_col = f'{metric}std'; anomaly_col_name = f'anomalia_{metric}' # Sin _ al final

        if median_col in df_merged.columns and std_col in df_merged.columns: df_merged[anomaly_col_name] = (df_merged[metric] - df_merged[median_col]) / (df_merged[std_col] + 1e-6)
        else: print(f"     [Clima] ADV: Faltan baselines {metric}. Anomalía 0."); df_merged[anomaly_col_name] = 0
        anomaly_cols.append(anomaly_col_name)

    df_per_comuna_anomalies = df_merged[['ts', 'comuna'] + anomaly_cols + expected_metrics].copy(); n_comunas = max(1, df_merged['comuna'].nunique())

    agg_functions = {};
    for col in anomaly_cols: agg_functions[col] = ['max', 'sum', lambda x, nc=n_comunas: (x > 2.5).sum() / nc if nc > 0 else 0]

    if not agg_functions: print("     [Clima] ADV: No se generaron funcs agregación."); df_agregado = pd.DataFrame({'ts': df_merged['ts'].unique()})
    else:
        df_agregado = df_merged.groupby('ts').agg(agg_functions).reset_index(); new_cols = ['ts'];
        for col in df_agregado.columns[1:]: agg_name = col[1] if col[1] != '<lambda_0>' else 'pct_comunas_afectadas'; new_cols.append(f"{col[0]}_{agg_name}")
        df_agregado.columns = new_cols

    print("     [Clima] Cálculo anomalías completado."); return df_agregado, df_per_comuna_anomalies


def generate_alerts_json(df_per_comuna, df_risk_proba, proba_threshold=0.5, impact_factor=100):
    print("     [Alertas] Generando 'alertas_climaticas.json'..."); df_alertas = pd.merge(df_per_comuna, df_risk_proba, on='ts', how='left')
    df_alertas = df_alertas[df_alertas['risk_proba'] > proba_threshold].copy()

    if df_alertas.empty: print("     [Alertas] No se detectaron alertas."); return []

    df_alertas['impacto_heuristico'] = (df_alertas['risk_proba'] - proba_threshold) * impact_factor; df_alertas = df_alertas.sort_values(['comuna', 'ts']); json_output = []

    for comuna, group in df_alertas.groupby('comuna'):
        if group.empty: continue
        group['time_diff'] = group['ts'].diff().dt.total_seconds() / 3600; group['bloque'] = (group['time_diff'] > 1).cumsum()

        for _, bloque_group in group.groupby('bloque'):
            ts_inicio = bloque_group['ts'].min(); ts_fin = bloque_group['ts'].max() + pd.Timedelta(minutes=59); anomalias_dict = {}
            for col in bloque_group.columns:
                if col.startswith('anomalia_'): anomalias_dict[f"{col.replace('anomalia_', '')}_z_max"] = round(bloque_group[col].max(), 2)
            alerta = {"comuna": comuna, "ts_inicio": ts_inicio.strftime('%Y-%m-%d %H:%M:%S'), "ts_fin": ts_fin.strftime('%Y-%m-%d %H:%M:%S'), "anomalias": anomalias_dict, "impacto_llamadas_adicionales": int(bloque_group['impacto_heuristico'].sum())}
            json_output.append(alerta)
    return json_output


# --- FUNCIÓN PRINCIPAL ORQUESTADORA ---

def main(horizonte_dias):
    print("="*60); print(f"INICIANDO PIPELINE DE INFERENCIA (v_main Modificado - Llamadas Iterativas)"); print(f"Zona Horaria: {TZ} | Horizonte: {horizonte_dias} días"); print("="*60)

    # --- 1. Cargar Modelos y Artefactos (v7) ---
    print("\n--- Fase 1: Cargando Modelos y Artefactos (v7) ---")
    try:
        model_planner = tf.keras.models.load_model(PLANNER_MODEL_FILE, compile=False) # Añadir compile=False
        scaler_planner = joblib.load(PLANNER_SCALER_FILE)
        cols_planner = _load_cols(PLANNER_COLS_FILE) # Usar helper

        model_risk = tf.keras.models.load_model(RISK_MODEL_FILE, compile=False) # Añadir compile=False
        scaler_risk = joblib.load(RISK_SCALER_FILE)
        try:
             cols_risk = _load_cols(RISK_COLS_FILE)
        except FileNotFoundError:
             print(f"     [Adv] {RISK_COLS_FILE} no encontrado."); cols_risk = []
        try:
             baselines_clima = joblib.load(RISK_BASELINES_FILE)
        except FileNotFoundError: print(f"     [Adv] {RISK_BASELINES_FILE} no encontrado."); baselines_clima = pd.DataFrame()

        model_tmo = tf.keras.models.load_model(TMO_MODEL_FILE, compile=False) # Añadir compile=False
        scaler_tmo = joblib.load(TMO_SCALER_FILE)
        cols_tmo = _load_cols(TMO_COLS_FILE) # Usar helper

        print("     [OK] Todos los modelos v7 cargados.")
    except Exception as e: print(f"     [ERROR] Falla crítica al cargar artefactos: {e}"); print("     Asegúrate que archivos v7 existan en 'models/'."); return

    # --- 2. Cargar Datos Históricos ---
    print("\n--- Fase 2: Cargando Datos Históricos ---")
    # --- INICIO CAMBIO: Usar ensure_ts y manejar errores ---
    try:
        df_hosting_full = read_data(HOSTING_FILE)
        df_hosting = ensure_ts(df_hosting_full) # Usa la función importada
    except Exception as e:
        print(f"     [ERROR] Falla crítica al cargar {HOSTING_FILE}: {e}"); return
    # --- FIN CAMBIO ---

    try:
        df_feriados_lookup = read_data(FERIADOS_FILE); df_feriados_lookup['Fecha_dt'] = pd.to_datetime(df_feriados_lookup['Fecha'], format='%d-%m-%Y', errors='coerce').dt.date
        feriados_list = set(df_feriados_lookup['Fecha_dt'].dropna())
    except Exception as e: print(f"     [Adv] No se pudo cargar {FERIADOS_FILE}. Error: {e}"); feriados_list = set()

    if 'feriados' not in df_hosting.columns: print("     [Info] Creando columna 'feriados'."); df_hosting['feriados'] = df_hosting.index.to_series().apply(lambda ts: _is_holiday(ts, feriados_list)) # Usar index y helper
    else: df_hosting['feriados'] = pd.to_numeric(df_hosting['feriados'], errors='coerce').fillna(0).astype(int)

    if 'recibidos' in df_hosting.columns and TARGET_CALLS not in df_hosting.columns: df_hosting = df_hosting.rename(columns={'recibidos': TARGET_CALLS})
    elif TARGET_CALLS not in df_hosting.columns: raise ValueError(f"No se encontró {TARGET_CALLS} ni 'recibidos'.")

    # --- INICIO CAMBIO: No agrupar aquí, usar historia completa ---
    # df_hosting_agg = df_hosting.groupby("ts").agg({TARGET_CALLS: 'sum', 'feriados': 'max'}).reset_index() # Solo agrupar feriados
    df_hosting_processed = df_hosting.copy() # Usar el DF ya indexado por ts
    if "es_dia_de_pago" not in df_hosting_processed.columns: # Asegurar día de pago
         df_hosting_processed['day'] = df_hosting_processed.index.day
         df_hosting_processed['es_dia_de_pago'] = df_hosting_processed['day'].isin(DIAS_DE_PAGO).astype(int)
    # Rellenar NaNs en columnas clave
    cols_to_ffill_calls = [TARGET_CALLS, "feriados", "es_dia_de_pago"]
    for c in cols_to_ffill_calls:
        if c in df_hosting_processed.columns:
            df_hosting_processed[c] = df_hosting_processed[c].ffill()
    df_hosting_processed = df_hosting_processed.dropna(subset=[TARGET_CALLS])
    # --- FIN CAMBIO ---

    # --- INICIO CAMBIO: Cargar TMO y manejar fallback ---
    df_tmo_hist = None
    try:
        df_tmo_hist_raw = read_data(TMO_FILE)
        df_tmo_hist = ensure_ts(df_tmo_hist_raw) # Usa la función importada
        df_tmo_hist.columns = [c.lower().strip().replace(' ', '_') for c in df_tmo_hist.columns]; # Normalizar columnas
        df_tmo_hist = df_tmo_hist.rename(columns={'tmo_general': TARGET_TMO}) # Renombrar target

        # Calcular proporciones (lógica de loader_tmo)
        if 'q_llamadas_comercial' in df_tmo_hist.columns and 'q_llamadas_general' in df_tmo_hist.columns:
            df_tmo_hist['proporcion_comercial'] = df_tmo_hist['q_llamadas_comercial'] / (df_tmo_hist['q_llamadas_general'] + 1e-6)
            df_tmo_hist['proporcion_tecnica'] = df_tmo_hist['q_llamadas_tecnico'] / (df_tmo_hist['q_llamadas_general'] + 1e-6)
        else:
             print("     [Adv] No se encontraron q_llamadas en TMO_HISTORICO. Proporciones serán 0.");
             df_tmo_hist['proporcion_comercial'] = 0.0
             df_tmo_hist['proporcion_tecnica'] = 0.0
        # Asegurar columnas tmo_comercial/tecnico si no existen
        if 'tmo_comercial' not in df_tmo_hist.columns: df_tmo_hist['tmo_comercial'] = np.nan
        if 'tmo_tecnico' not in df_tmo_hist.columns: df_tmo_hist['tmo_tecnico'] = np.nan

        # Ffill TMO
        tmo_static_features_list = ["proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]
        cols_to_ffill_tmo = [TARGET_TMO] + tmo_static_features_list
        for c in cols_to_ffill_tmo:
            if c in df_tmo_hist.columns:
                 df_tmo_hist[c] = df_tmo_hist[c].ffill()

        print("     [OK] TMO_HISTORICO.csv cargado y procesado.")

    except FileNotFoundError:
        print(f"     [Adv] {TMO_FILE} no encontrado. Se usará fallback desde historical_data.")
        df_tmo_hist = None # Forzar fallback
    except Exception as e:
        print(f"     [Adv] Error cargando {TMO_FILE}: {e}. Se usará fallback.")
        df_tmo_hist = None # Forzar fallback

    if df_tmo_hist is None: # Si falló la carga, usar fallback
        is_fallback = True
        df_tmo_hist = df_hosting_processed.copy() # Usar datos de hosting
        # Asegurar que las columnas existan, aunque sean NaN
        tmo_static_features_list = ["proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]
        if TARGET_TMO not in df_tmo_hist.columns: df_tmo_hist[TARGET_TMO] = np.nan
        for c in tmo_static_features_list:
            if c not in df_tmo_hist.columns: df_tmo_hist[c] = np.nan
        # Ffill de nuevo por si acaso
        cols_to_ffill_tmo_fallback = [TARGET_TMO] + tmo_static_features_list
        for c in cols_to_ffill_tmo_fallback:
            if c in df_tmo_hist.columns:
                 df_tmo_hist[c] = df_tmo_hist[c].ffill()
    else:
        is_fallback = False
    # --- FIN CAMBIO ---

    last_hist_ts = df_hosting_processed.index.max(); print(f"     [OK] Datos históricos listos. Último timestamp: {last_hist_ts}")


    # --- 3. Generar Esqueleto Futuro ---
    print("\n--- Fase 3: Generando Esqueleto de Fechas Futuras ---")
    start_future = last_hist_ts + pd.Timedelta(hours=1); end_future = start_future + pd.Timedelta(days=horizonte_dias, hours=-1) # Ajuste para incluir última hora
    future_ts_index = pd.date_range(start=start_future, end=end_future, freq='h', tz=TZ)
    df_future = pd.DataFrame(index=future_ts_index)
    df_future.index.name = 'ts'

    # --- INICIO CAMBIO: Usar add_time_parts y helpers ---
    df_future = add_time_parts(df_future) # add_time_parts importado
    df_future['feriados'] = df_future.index.to_series().apply(lambda ts: _is_holiday(ts, feriados_list))
    # 'es_dia_de_pago' se calculará en el bucle/vector
    # --- FIN CAMBIO ---

    print(f"     [OK] Esqueleto futuro creado: {df_future.index.min()} a {df_future.index.max()}")


    # --- 4. Pipeline Clima ---
    print("\n--- Fase 4: Pipeline de Clima (Analista de Riesgos) ---")
    df_weather_future_raw = fetch_future_weather(start_future, end_future)
    df_agg_anomalies, df_per_comuna_anomalies = process_future_climate(df_weather_future_raw, baselines_clima if not baselines_clima.empty else pd.DataFrame())

    # --- INICIO CAMBIO: Merge con índice ---
    df_future = df_future.merge(df_agg_anomalies, left_index=True, right_on='ts', how='left')
    df_future = df_future.set_index('ts') # Volver a poner ts como índice
    # --- FIN CAMBIO ---

    # --- INICIO CAMBIO: Ffill después del merge ---
    numeric_cols_future = df_future.select_dtypes(include=np.number).columns
    # Llenar NaNs de clima con 0 (sin anomalía) en lugar de media
    clima_cols_in_future = [c for c in df_agg_anomalies.columns if c != 'ts' and c in df_future.columns]
    df_future[clima_cols_in_future] = df_future[clima_cols_in_future].fillna(0)
    # Rellenar cualquier otro NaN numérico con 0 (ej. si falla carga de algo)
    df_future[numeric_cols_future] = df_future[numeric_cols_future].fillna(0)
    # --- FIN CAMBIO ---


    if cols_risk and all(c in df_future.columns for c in cols_risk):
        # --- INICIO CAMBIO: dummies_and_reindex ---
        # Asegurar dummies de tiempo si el modelo de riesgo las necesita
        # Nota: El entrenamiento v7 original NO usa dummies de tiempo para riesgo
        X_risk = dummies_and_reindex(df_future.copy(), cols_risk) # Usar helper
        # --- FIN CAMBIO ---
        X_risk_s = scaler_risk.transform(X_risk)
        df_future['risk_proba'] = model_risk.predict(X_risk_s); print("     [OK] Predicciones riesgo generadas.")
    else: print("     [Adv] Faltan columnas/config risk. 'risk_proba'=0."); df_future['risk_proba'] = 0.0

    df_risk_proba_output = df_future[['risk_proba']].reset_index(); # Reset index para merge
    alertas_json_data = generate_alerts_json(df_per_comuna_anomalies, df_risk_proba_output)


# ==============================================================================
# === INICIO BLOQUE MODIFICADO (FASE 5 - LLAMADAS ITERATIVAS) ===
# ==============================================================================
    print("\n--- Fase 5: Pipeline de Llamadas (Iterativo) ---")

    # Preparar historia reciente para el bucle (dfp_full)
    start_hist_calls = last_hist_ts - pd.Timedelta(days=HIST_WINDOW_DAYS)
    df_recent_calls = df_hosting_processed.loc[df_hosting_processed.index >= start_hist_calls].copy()
    if df_recent_calls.empty:
        df_recent_calls = df_hosting_processed.copy()

    cols_iter_calls = [TARGET_CALLS]
    if "feriados" in df_recent_calls.columns:
        cols_iter_calls.append("feriados")
    if "es_dia_de_pago" in df_recent_calls.columns:
        cols_iter_calls.append("es_dia_de_pago")

    dfp_calls = df_recent_calls[cols_iter_calls].copy()
    dfp_full = dfp_calls.copy() # dfp_full solo tendrá llamadas

    dfp_full[TARGET_CALLS] = dfp_full[TARGET_CALLS].ffill().fillna(0.0)
    if "es_dia_de_pago" in dfp_full.columns:
         dfp_full["es_dia_de_pago"] = dfp_full["es_dia_de_pago"].ffill().fillna(0)

    print("     Iniciando predicción iterativa (SÓLO Llamadas)...")
    for ts in df_future.index: # Iterar sobre el índice del esqueleto futuro
        # 1. Crear el 'slice' temporal para esta hora
        # Usamos dfp_full que contiene la historia + predicciones anteriores
        tmp = pd.concat([dfp_full, pd.DataFrame(index=[ts])])

        # 2. ffill de los targets
        tmp[TARGET_CALLS] = tmp[TARGET_CALLS].ffill()

        # 3. Añadir feriado y día de pago para esta hora 'ts'
        if "feriados" in tmp.columns:
            tmp.loc[ts, "feriados"] = _is_holiday(ts, feriados_list)

        if "es_dia_de_pago" in tmp.columns:
            tmp.loc[ts, "es_dia_de_pago"] = _is_payday(ts)

        # 4. Crear features (Lags, MAs de LLAMADAS, Tiempo)
        tmp_with_feats = add_lags_mas(tmp, TARGET_CALLS) # Lags/MAs de llamadas
        tmp_with_feats = add_time_parts(tmp_with_feats) # Tiempo (dow, hour, etc.)

        # 5. PREDECIR LLAMADAS (PLANNER) para esta hora 'ts'
        current_row = tmp_with_feats.tail(1)

        # Usar helper para dummies y reindex
        X_pl = dummies_and_reindex(current_row, cols_planner)
        yhat_calls = float(model_planner.predict(scaler_planner.transform(X_pl), verbose=0).flatten()[0])
        yhat_calls = max(0.0, yhat_calls) # Asegurar no negativo

        # 6. Guardar predicción de llamadas en dfp_full para la siguiente iteración
        dfp_full.loc[ts, TARGET_CALLS] = yhat_calls

        # 7. Guardar también feriado/dia_pago (ya estaba calculado en tmp)
        if "feriados" in dfp_full.columns:
            dfp_full.loc[ts, "feriados"] = tmp.loc[ts, "feriados"]
        if "es_dia_de_pago" in dfp_full.columns:
            dfp_full.loc[ts, "es_dia_de_pago"] = tmp.loc[ts, "es_dia_de_pago"]

    print("     [OK] Predicción iterativa de llamadas completada.")

    # Ahora dfp_full contiene la historia + las predicciones iterativas de llamadas
    # Seleccionamos solo el futuro para el resultado final de llamadas
    df_future['llamadas_hora'] = np.round(dfp_full.loc[df_future.index, TARGET_CALLS]).astype(int)

# ==============================================================================
# === FIN BLOQUE MODIFICADO ===
# ==============================================================================


    # --- 6. Pipeline TMO (Vectorizado - Lógica Prototipo v_main) ---
    print("\n--- Fase 6: Pipeline de TMO (Vectorizado - Prototipo v_main) ---")

    # --- INICIO CAMBIO: Usar lógica de v14.1 para last_tmo_data_static ---
    tmo_static_features_list = ["proporcion_comercial", "proporcion_tecnica", "tmo_comercial", "tmo_tecnico"]
    if df_tmo_hist.empty:
        print("     [Adv] Historial TMO vacío. Usando 0.0 para features estáticas.")
        last_tmo_data_static = {c: 0.0 for c in tmo_static_features_list}
    else:
        # Asegurar que las columnas existan antes de indexar
        valid_static_features = [c for c in tmo_static_features_list if c in df_tmo_hist.columns]
        # Usar .iloc[-1] para obtener la última fila como Series
        last_row_series = df_tmo_hist.iloc[-1]
        last_tmo_data_static = last_row_series[valid_static_features].to_dict()

        # Añadir las que faltaron (si alguna) con 0.0
        for c in tmo_static_features_list:
            if c not in last_tmo_data_static:
                last_tmo_data_static[c] = 0.0
                print(f"     [Debug] Columna estática faltante '{c}' rellenada con 0.")
            elif pd.isna(last_tmo_data_static[c]):
                last_tmo_data_static[c] = 0.0
                print(f"     [Debug] Columna estática NaN '{c}' rellenada con 0.")

    print(f"     [Info] Usando valores TMO estáticos (prototipo v_main): {last_tmo_data_static}")
    # --- FIN CAMBIO ---


    # Preparar DataFrame para predicción TMO
    df_tmo_features_future = df_future.copy() # Ya tiene llamadas_hora, feriados, risk_proba, etc.
    df_tmo_features_future[TARGET_CALLS] = df_tmo_features_future['llamadas_hora'] # Renombrar para el modelo TMO

    # Añadir features de tiempo (si no existen, add_time_parts ya lo hizo antes)
    if not all (c in df_tmo_features_future.columns for c in ['dow', 'month', 'hour', 'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow']):
        df_tmo_features_future = add_time_parts(df_tmo_features_future)

    # Añadir 'es_dia_de_pago' al futuro
    if 'day' not in df_tmo_features_future.columns:
        df_tmo_features_future['day'] = df_tmo_features_future.index.day
    df_tmo_features_future['es_dia_de_pago'] = df_tmo_features_future['day'].isin(DIAS_DE_PAGO).astype(int)


    # Añadir las features ESTÁTICAS (lógica prototipo v_main)
    for col_name, static_value in last_tmo_data_static.items():
        df_tmo_features_future[col_name] = static_value

    # Dummies y Reindex (lógica prototipo v_main)
    # Esto creará las columnas autorregresivas (lag_tmo_...) y las rellenará con 0
    X_tmo = dummies_and_reindex(df_tmo_features_future, cols_tmo) # Usar helper

    # Escalar y Predecir (lógica prototipo v_main)
    X_tmo_s = scaler_tmo.transform(X_tmo)
    yhat_tmo_vector = model_tmo.predict(X_tmo_s, verbose=0).flatten()

    # Guardar TMO en el DataFrame futuro
    df_future['tmo_hora'] = np.round(np.clip(yhat_tmo_vector, 0, None)).astype(int)

    print("     [OK] Predicciones TMO (Vectorizado) generadas.")


    # --- 7. Generar Salidas Finales ---
    print("\n--- Fase 7: Generando Archivos JSON de Salida ---")
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    # --- INICIO CAMBIO: Usar required_agents importado ---
    df_future['agents_prod'] = 0
    for ts_idx in df_future.index: # Iterar por índice para asegurar alineación
         calls = float(df_future.loc[ts_idx, "llamadas_hora"])
         tmo = float(df_future.loc[ts_idx, "tmo_hora"])
         # Asegurar tmo > 0 para evitar errores en erlang
         tmo = max(tmo, 1.0) if calls > 0 else tmo
         agents, _ = required_agents(calls, tmo)
         df_future.loc[ts_idx, "agents_prod"] = int(agents)

    df_future['agentes_requeridos'] = df_future["agents_prod"].apply(schedule_agents) # 'schedule_agents' es el shrinkage
    # --- FIN CAMBIO ---


    df_horaria = df_future[['llamadas_hora', 'tmo_hora', 'agentes_requeridos']].reset_index() # reset_index para tener 'ts'
    df_horaria['ts'] = df_horaria['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

    output_path_horaria = os.path.join(PUBLIC_DIR, "prediccion_horaria.json")
    # --- INICIO CAMBIO: Usar indent=2 y ensure_ascii=False ---
    df_horaria.to_json(output_path_horaria, orient='records', indent=2, force_ascii=False)
    # --- FIN CAMBIO ---
    print(f"     [OK] Archivo 'prediccion_horaria.json' guardado.")

    # --- INICIO CAMBIO: Simplificar cálculo diario y nombre archivo ---
    df_horaria_para_diaria = df_future.copy() # df_future tiene índice ts
    df_horaria_para_diaria['fecha'] = df_horaria_para_diaria.index.date.astype(str)

    # Calcular TMO diario ponderado
    df_horaria_para_diaria['vol_tmo'] = df_horaria_para_diaria['llamadas_hora'] * df_horaria_para_diaria['tmo_hora']
    df_diaria_agg = df_horaria_para_diaria.groupby('fecha').agg(
        llamadas_diarias=('llamadas_hora', 'sum'),
        vol_tmo_total=('vol_tmo', 'sum')
    )
    # Evitar división por cero
    df_diaria_agg['tmo_diario'] = np.where(
        df_diaria_agg['llamadas_diarias'] > 0,
        df_diaria_agg['vol_tmo_total'] / df_diaria_agg['llamadas_diarias'],
        0 # O podrías usar la media simple como fallback si prefieres
    )
    # Fallback con media simple si llamadas son 0 (copiado de tu original)
    if (df_diaria_agg['llamadas_diarias'] == 0).any():
        tmo_simple_avg = df_horaria_para_diaria.groupby('fecha')['tmo_hora'].mean().fillna(0)
        df_diaria_agg['tmo_diario'] = df_diaria_agg['tmo_diario'].where(df_diaria_agg['llamadas_diarias'] > 0, tmo_simple_avg).fillna(0)


    df_diaria_final = df_diaria_agg.reset_index()[['fecha', 'llamadas_diarias', 'tmo_diario']]
    df_diaria_final['llamadas_diarias'] = df_diaria_final['llamadas_diarias'].astype(int)
    df_diaria_final['tmo_diario'] = df_diaria_final['tmo_diario'].round(2) # Redondear TMO diario

    output_path_diaria = os.path.join(PUBLIC_DIR, "prediccion_diaria.json") # Nombre estándar
    df_diaria_final.to_json(output_path_diaria, orient='records', indent=2, force_ascii=False)
    # --- FIN CAMBIO ---
    print(f"     [OK] Archivo 'prediccion_diaria.json' guardado.")

    output_path_alertas = os.path.join(PUBLIC_DIR, "alertas_climaticas.json")
    with open(output_path_alertas, 'w', encoding='utf-8') as f: json.dump(alertas_json_data, f, indent=2, ensure_ascii=False)
    print(f"     [OK] Archivo 'alertas_climaticas.json' guardado.")

    print("\n" + "="*60); print("PIPELINE DE INFERENCIA COMPLETADO EXITOSAMENTE."); print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Inferencia IA (v_main Modificado).")
    parser.add_argument("--horizonte", type=int, default=120, help="Horizonte predicción días")
    args = parser.parse_args()
    main(horizonte_dias=args.horizonte)