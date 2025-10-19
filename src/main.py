# (Imports y configuración global - ajustados para v24.1)
import os
import argparse
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# --- CONFIGURACIÓN GLOBAL ---
TZ = 'America/Santiago'; os.environ['TZ'] = TZ
BASE_DIR = os.path.dirname(os.path.abspath(__file__)); ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data"); MODEL_DIR = os.path.join(ROOT_DIR, "models"); PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
# Archivos v24.1
PLANNER_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_planner.keras"); PLANNER_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_planner.pkl"); PLANNER_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_planner.json")
PLANNER_BASELINES_FILE = os.path.join(MODEL_DIR, "baselines_llamadas.pkl") # <-- Nuevo v24
RISK_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_riesgos.keras"); RISK_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_riesgos.pkl"); RISK_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_riesgos.json"); RISK_BASELINES_FILE = os.path.join(MODEL_DIR, "baselines_clima.pkl")
TMO_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_tmo.keras"); TMO_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_tmo.pkl"); TMO_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_tmo.json")
HOSTING_FILE = os.path.join(DATA_DIR, "historical_data.csv"); TMO_FILE = os.path.join(DATA_DIR, "TMO_HISTORICO.csv"); FERIADOS_FILE = os.path.join(DATA_DIR, "Feriados_Chilev2.csv"); CLIMA_HIST_FILE = os.path.join(DATA_DIR, "historical_data.csv")
TARGET_CALLS = "recibidos_nacional"; TARGET_TMO = "tmo_general"
QUANTILE_P = 0.50 # Cuantil usado en entreno v24
N_STEPS = 24
K_MAD_WEEKDAY = 5.0; K_MAD_WEEKEND = 6.5 # Parámetros Capping
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'; warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl'); warnings.filterwarnings('ignore', category=FutureWarning)

def quantile_loss(q):
    def loss(y_true, y_pred): e = y_true - y_pred; return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
    return loss

# --- FUNCIONES DE UTILIDAD ---
# (read_data, ensure_ts_and_tz, add_time_parts - sin cambios)
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

def ensure_ts_and_tz(df):
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]; date_col = next((c for c in df.columns if 'fecha' in c), None); hour_col = next((c for c in df.columns if 'hora' in c), None)
    if not date_col or not hour_col: raise ValueError("No se encontraron 'fecha' y 'hora'.")
    try: df["ts"] = pd.to_datetime(df[date_col] + ' ' + df[hour_col], format='%d-%m-%Y %H:%M:%S', errors='raise')
    except (ValueError, TypeError): print(f"  [Adv] Formato dd-mm-yyyy no detectado. Intentando inferir."); df["ts"] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[hour_col].astype(str), errors='coerce')
    df = df.dropna(subset=["ts"])
    if df["ts"].dt.tz is None: df["ts"] = df["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    else: df["ts"] = df["ts"].dt.tz_convert(TZ)
    df = df.dropna(subset=["ts"])
    return df.sort_values("ts")

def add_time_parts(df):
    df_copy = df.copy(); df_copy["dow"] = df_copy["ts"].dt.dayofweek; df_copy["month"] = df_copy["ts"].dt.month; df_copy["hour"] = df_copy["ts"].dt.hour; df_copy["day"] = df_copy["ts"].dt.day
    df_copy["semana_del_mes"] = (df_copy["day"] - 1) // 7 + 1; df_copy["es_dia_de_pago"] = df_copy['day'].isin([1, 2, 15, 16, 29, 30, 31]).astype(int)
    df_copy["es_domingo"] = (df_copy["dow"] == 6).astype(int); df_copy["es_madrugada"] = (df_copy["hour"] < 6).astype(int)
    df_copy["es_navidad"] = ((df_copy["month"] == 12) & (df_copy["day"] == 25)).astype(int); df_copy["es_ano_nuevo"] = ((df_copy["month"] == 1) & (df_copy["day"] == 1)).astype(int)
    df_copy["es_fiestas_patrias"] = ((df_copy["month"] == 9) & (df_copy["day"].isin([18, 19]))).astype(int)
    dias_quincena = [15, 16, 29, 30, 31, 1, 2]; df_copy["es_quincena"] = df_copy["day"].isin(dias_quincena).astype(int)
    df_copy["sin_hour"] = np.sin(2 * np.pi * df_copy["hour"] / 24); df_copy["cos_hour"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
    df_copy["sin_dow"] = np.sin(2 * np.pi * df_copy["dow"] / 7); df_copy["cos_dow"] = np.cos(2 * np.pi * df_copy["dow"] / 7)
    return df_copy

# (normalize_climate_columns, calculate_erlang_agents, create_inference_sequences - sin cambios)
def normalize_climate_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {'temperatura': ['temperature_2m', 'temperatura', 'temp', 'temp_2m'], 'precipitacion': ['precipitation', 'precipitacion', 'precipitación', 'rain_mm', 'rain'], 'lluvia': ['rain', 'lluvia', 'rainfall']}
    df_renamed = df.copy(); df_renamed.columns = [c.lower().strip().replace(' ', '_') for c in df_renamed.columns]
    for std, poss in column_map.items():
        for name in poss:
            if name in df_renamed.columns: df_renamed.rename(columns={name: std}, inplace=True); break
    return df_renamed
def calculate_erlang_agents(calls_per_hour, tmo_seconds, occupancy_target=0.85):
    calls = pd.to_numeric(calls_per_hour, errors='coerce').fillna(0); tmo = pd.to_numeric(tmo_seconds, errors='coerce').fillna(0)
    if (calls.sum() == 0) or (tmo <= 0).all(): return pd.Series(0, index=calls_per_hour.index)
    tmo_safe = tmo.replace(0, 1e-6); traffic_intensity = (calls * tmo_safe) / 3600
    agents = np.ceil(traffic_intensity / occupancy_target); agents[calls > 0] = agents[calls > 0].apply(lambda x: max(x, 1))
    agents = agents.replace([np.inf, -np.inf], np.nan).fillna(0)
    return agents.astype(int)
def create_inference_sequences(historical_scaled_context, future_scaled_data, time_steps=N_STEPS):
    n_future_steps = future_scaled_data.shape[0]; n_features = future_scaled_data.shape[1]
    combined_data = np.vstack((historical_scaled_context, future_scaled_data))
    Xs = []
    for i in range(n_future_steps): start_index = i; end_index = start_index + time_steps; window = combined_data[start_index:end_index]; Xs.append(window)
    if not Xs: return np.empty((0, time_steps, n_features))
    return np.array(Xs)

# (aplicar_capping_mad - sin cambios v24)
def aplicar_capping_mad(df_predicciones, df_baselines, k_weekday=K_MAD_WEEKDAY, k_weekend=K_MAD_WEEKEND):
    print("  Aplicando capping robusto (MAD) post-procesamiento...")
    df = df_predicciones.copy();
    if df_baselines.empty: print("    [Advertencia] No baselines. No capping."); return df
    if 'dow' not in df.columns or 'hour' not in df.columns: print("    [Error] Faltan 'dow'/'hour'. No capping."); return df
    df_merged = pd.merge(df, df_baselines, on=['dow', 'hour'], how='left')
    df_merged['mediana'].fillna(df_baselines['mediana'].mean(), inplace=True); df_merged['mad'].fillna(df_baselines['mad'].mean(), inplace=True); df_merged['q95'].fillna(df_baselines['q95'].mean(), inplace=True)
    K = np.where(df_merged['dow'].isin([5, 6]), k_weekend, k_weekday); upper_cap = df_merged['mediana'] + K * df_merged['mad']
    mask = (df_merged['llamadas_hora'] > upper_cap) & (df_merged['llamadas_hora'] > df_merged['q95'])
    n_capped = mask.sum()
    if n_capped > 0: print(f"    - Recortando {n_capped} predicciones."); df_merged.loc[mask, 'llamadas_hora'] = upper_cap[mask]
    df_result = df_merged[df_predicciones.columns].copy(); df_result['llamadas_hora'] = df_result['llamadas_hora'].round().clip(0).astype(int)
    print(f"  Capping completado."); return df_result

# --- FUNCIONES DEL PIPELINE DE INFERENCIA ---
# (fetch_future_weather, process_future_climate, generate_alerts_json - sin cambios)
def fetch_future_weather(start_date, end_date):
    print("    [Clima] SIMULANDO API de clima futuro...")
    try: df_hist = read_data(CLIMA_HIST_FILE)
    except FileNotFoundError: print(f"    [Clima] ADVERTENCIA: {CLIMA_HIST_FILE} no encontrado. Dummy."); comunas = ['Santiago']; dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts'])); df_simulado['temperatura'] = 15; df_simulado['precipitacion'] = 0; df_simulado['lluvia'] = 0; return df_simulado.reset_index()
    df_hist = ensure_ts_and_tz(df_hist); df_hist = normalize_climate_columns(df_hist); climate_cols_found = [col for col in ['temperatura', 'precipitacion', 'lluvia'] if col in df_hist.columns]
    if not climate_cols_found: print(f"    [Clima] ADVERTENCIA: No cols clima en {CLIMA_HIST_FILE}. Dummy."); comunas = ['Santiago']; dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts'])); df_simulado['temperatura'] = 15; df_simulado['precipitacion'] = 0; df_simulado['lluvia'] = 0; return df_simulado.reset_index()
    if 'comuna' not in df_hist.columns: print("    [Clima] ADVERTENCIA: 'comuna' no encontrada. Dummy 'Santiago'."); df_hist['comuna'] = 'Santiago'
    future_dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_future_list = []
    for date in future_dates:
        try: sim_date = date.replace(year=date.year - 1); data_sim = df_hist[df_hist['ts'] == sim_date];
        except ValueError: sim_date = date - pd.Timedelta(days=365); data_sim = df_hist[df_hist['ts'] == sim_date] # Approx for leap year
        if not data_sim.empty: data_sim['ts'] = date; df_future_list.append(data_sim)
    if not df_future_list:
        print("    [Clima] No match año anterior. Usando última semana."); last_week = df_hist[df_hist['ts'] >= df_hist['ts'].max() - pd.Timedelta(days=7)]
        if last_week.empty: print("    [Clima] ADVERTENCIA: No datos última semana. Dummy."); return fetch_future_weather(start_date, end_date)
        last_week_mapping = last_week.set_index('ts');
        for date in future_dates: sim_date = date - pd.Timedelta(days=7); sim_ts_floor = sim_date.floor('h')
        if sim_ts_floor in last_week_mapping.index: data_sim = last_week_mapping.loc[[sim_ts_floor]].reset_index(drop=True); data_sim['ts'] = date; df_future_list.append(data_sim)
    if not df_future_list: print("    [Clima] ADVERTENCIA: Falló simulación. Dummy."); return fetch_future_weather(start_date, end_date)
    df_simulado = pd.concat(df_future_list); all_comunas = df_hist['comuna'].unique(); all_dates = future_dates; full_index = pd.MultiIndex.from_product([all_comunas, all_dates], names=['comuna', 'ts'])
    df_final = df_simulado.set_index(['comuna', 'ts']).reindex(full_index); df_final = df_final.groupby(level='comuna').ffill().bfill(); df_final = df_final.fillna(0);
    print(f"    [Clima] Simulación API completada. {len(df_final)} registros."); return df_final.reset_index()

def process_future_climate(df_future_weather, df_baselines):
    print("    [Clima] Procesando datos futuros y calculando anomalías..."); df = normalize_climate_columns(df_future_weather.copy())
    if 'ts' not in df.columns or df['ts'].isnull().all(): df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    if df["ts"].dt.tz is None: df["ts"] = df["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    else: df["ts"] = df["ts"].dt.tz_convert(TZ)
    df = df.dropna(subset=['ts']).sort_values(['comuna', 'ts']); df['dow'] = df['ts'].dt.dayofweek; df['hour'] = df['ts'].dt.hour
    if df_baselines.empty: print("    [Clima] Adv: Baselines vacíos, anomalías serán 0."); df_merged = df.copy(); df_merged['temperatura_median'] = 0; df_merged['temperatura_std'] = 1 # Etc
    else: df_merged = pd.merge(df, df_baselines, on=['comuna', 'dow', 'hour'], how='left') # v14.1 fix
    numeric_cols = df_merged.select_dtypes(include=np.number).columns; df_merged[numeric_cols] = df_merged[numeric_cols].fillna(df_merged[numeric_cols].mean())
    expected_metrics = [c for c in ['temperatura', 'precipitacion', 'lluvia'] if c in df.columns]; anomaly_cols = []
    for metric in expected_metrics:
        median_col = f'{metric}_median'; std_col = f'{metric}_std'; anomaly_col_name = f'anomalia_{metric}'
        if median_col in df_merged.columns and std_col in df_merged.columns: df_merged[anomaly_col_name] = (df_merged[metric] - df_merged[median_col]) / (df_merged[std_col] + 1e-6)
        else: print(f"    [Clima] ADV: Faltan baselines {metric}. Anomalía 0."); df_merged[anomaly_col_name] = 0
        anomaly_cols.append(anomaly_col_name)
    df_per_comuna_anomalies = df_merged[['ts', 'comuna'] + anomaly_cols + expected_metrics].copy(); n_comunas = max(1, df_merged['comuna'].nunique())
    agg_functions = {};
    for col in anomaly_cols: agg_functions[col] = ['max', 'sum', lambda x, nc=n_comunas: (x > 2.5).sum() / nc if nc > 0 else 0]
    if not agg_functions: print("    [Clima] ADV: No se generaron funcs agregación."); df_agregado = pd.DataFrame({'ts': df_merged['ts'].unique()})
    else:
        df_agregado = df_merged.groupby('ts').agg(agg_functions).reset_index(); new_cols = ['ts'];
        for col in df_agregado.columns[1:]: agg_name = col[1] if col[1] != '<lambda_0>' else 'pct_comunas_afectadas'; new_cols.append(f"{col[0]}_{agg_name}")
        df_agregado.columns = new_cols
    print("    [Clima] Cálculo anomalías completado."); return df_agregado, df_per_comuna_anomalies

def generate_alerts_json(df_per_comuna, df_risk_proba, proba_threshold=0.5, impact_factor=100):
    print("    [Alertas] Generando 'alertas_climaticas.json'..."); df_alertas = pd.merge(df_per_comuna, df_risk_proba, on='ts', how='left')
    df_alertas = df_alertas[df_alertas['risk_proba'] > proba_threshold].copy()
    if df_alertas.empty: print("    [Alertas] No se detectaron alertas."); return []
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
    print("="*60); print(f"INICIANDO PIPELINE DE INFERENCIA (v_main 24.1 - LSTM Base + Capping MAD)"); print(f"Zona Horaria: {TZ} | Horizonte: {horizonte_dias} días"); print("="*60) # v24.1

    # --- 1. Cargar Modelos y Artefactos (v24.1) ---
    print("\n--- Fase 1: Cargando Modelos y Artefactos (v24.1 - Base q=0.50 + Baselines) ---")
    try:
        custom_objects_dict = {'loss': quantile_loss(q=QUANTILE_P)} # QUANTILE_P = 0.50
        model_planner = tf.keras.models.load_model(PLANNER_MODEL_FILE, custom_objects=custom_objects_dict); scaler_planner = joblib.load(PLANNER_SCALER_FILE)
        with open(PLANNER_COLS_FILE, 'r') as f: cols_planner = json.load(f)
        try: baselines_llamadas = joblib.load(PLANNER_BASELINES_FILE)
        except FileNotFoundError: print(f"  [ERROR Crítico] {PLANNER_BASELINES_FILE} no encontrado."); return
        model_risk = tf.keras.models.load_model(RISK_MODEL_FILE); scaler_risk = joblib.load(RISK_SCALER_FILE)
        # --- AJUSTE v24.1: Corrección indentación ---
        try:
            with open(RISK_COLS_FILE, 'r') as f: cols_risk = json.load(f)
        except FileNotFoundError: # <-- Indentación Correcta
            print(f"  [Adv] {RISK_COLS_FILE} no encontrado."); cols_risk = []
        # --- Fin Ajuste ---
        try: baselines_clima = joblib.load(RISK_BASELINES_FILE)
        except FileNotFoundError: print(f"  [Adv] {RISK_BASELINES_FILE} no encontrado."); baselines_clima = pd.DataFrame()
        model_tmo = tf.keras.models.load_model(TMO_MODEL_FILE, custom_objects=custom_objects_dict); scaler_tmo = joblib.load(TMO_SCALER_FILE)
        with open(TMO_COLS_FILE, 'r') as f: cols_tmo = json.load(f)
        print("  [OK] Todos los modelos v24.1 y baselines cargados.")
    except Exception as e: print(f"  [ERROR] Falla crítica al cargar artefactos: {e}"); print("  Asegúrate que archivos v24.1 existan en 'models/'."); return

    # --- 2. Cargar Datos Históricos ---
    print("\n--- Fase 2: Cargando Datos Históricos ---")
    # (Código sin cambios)
    df_hosting_full = read_data(HOSTING_FILE); df_hosting = ensure_ts_and_tz(df_hosting_full)
    try:
        df_feriados_lookup = read_data(FERIADOS_FILE); df_feriados_lookup['Fecha_dt'] = pd.to_datetime(df_feriados_lookup['Fecha'], format='%d-%m-%Y', errors='coerce').dt.date
        feriados_list = set(df_feriados_lookup['Fecha_dt'].dropna())
    except Exception as e: print(f"  [Adv] No se pudo cargar {FERIADOS_FILE}. Error: {e}"); feriados_list = set()
    if 'feriados' not in df_hosting.columns: print("  [Info] Creando columna 'feriados'."); df_hosting['feriados'] = df_hosting['ts'].dt.date.isin(feriados_list).astype(int)
    else: df_hosting['feriados'] = pd.to_numeric(df_hosting['feriados'], errors='coerce').fillna(0).astype(int)
    df_hosting['dia_despues_feriado'] = df_hosting['feriados'].shift(24).fillna(0).astype(int)
    df_hosting['dia_antes_feriado'] = df_hosting['feriados'].shift(-24).fillna(0).astype(int)
    if 'recibidos' in df_hosting.columns and TARGET_CALLS not in df_hosting.columns: df_hosting = df_hosting.rename(columns={'recibidos': TARGET_CALLS})
    elif TARGET_CALLS not in df_hosting.columns: raise ValueError(f"No se encontró {TARGET_CALLS} ni 'recibidos'.")
    df_hosting_agg = df_hosting.groupby("ts").agg({TARGET_CALLS: 'sum', 'feriados': 'max', 'dia_despues_feriado': 'max', 'dia_antes_feriado':'max'}).reset_index()
    df_hosting_processed = add_time_parts(df_hosting_agg)
    hist_features_planner = ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'feriados', 'dia_despues_feriado', 'dia_antes_feriado', 'es_quincena', 'es_dia_de_pago', 'month', 'semana_del_mes', 'es_domingo', 'es_madrugada', 'es_navidad', 'es_ano_nuevo', 'es_fiestas_patrias', TARGET_CALLS]
    X_hist_df = pd.get_dummies(df_hosting_processed[hist_features_planner], columns=['month', 'semana_del_mes'])
    X_hist_df = X_hist_df.reindex(columns=cols_planner, fill_value=0)
    X_hist_scaled = scaler_planner.transform(X_hist_df)
    if len(X_hist_scaled) < N_STEPS: raise ValueError(f"Datos históricos insuficientes ({len(X_hist_scaled)}) para contexto LSTM ({N_STEPS})")
    historical_context_planner = X_hist_scaled[-N_STEPS:]
    df_tmo_hist = read_data(TMO_FILE); df_tmo_hist = ensure_ts_and_tz(df_tmo_hist); df_tmo_hist.columns = [c.lower().strip().replace(' ', '_') for c in df_tmo_hist.columns]; df_tmo_hist = df_tmo_hist.rename(columns={'tmo_general': TARGET_TMO})
    if TARGET_TMO not in df_tmo_hist.columns and all(c in df_tmo_hist.columns for c in ['tmo_comercial', 'q_comercial', 'tmo_tecnico', 'q_tecnico', 'q_general']): df_tmo_hist[TARGET_TMO] = (df_tmo_hist['tmo_comercial'] * df_tmo_hist['q_comercial'] + df_tmo_hist['tmo_tecnico'] * df_tmo_hist['q_tecnico']) / (df_tmo_hist['q_general'] + 1e-6)
    if 'q_llamadas_comercial' in df_tmo_hist.columns and 'q_llamadas_general' in df_tmo_hist.columns: df_tmo_hist['proporcion_comercial'] = df_tmo_hist['q_llamadas_comercial'] / (df_tmo_hist['q_llamadas_general'] + 1e-6); df_tmo_hist['proporcion_tecnica'] = df_tmo_hist['q_llamadas_tecnico'] / (df_tmo_hist['q_llamadas_general'] + 1e-6)
    else: print("  [Adv] No cols q_llamadas TMO."); df_tmo_hist['proporcion_comercial'] = 0; df_tmo_hist['proporcion_tecnica'] = 0
    last_hist_ts = df_hosting_processed['ts'].max(); print(f"  [OK] Datos históricos cargados. Último timestamp: {last_hist_ts}")

    # --- 3. Generar Esqueleto Futuro ---
    print("\n--- Fase 3: Generando Esqueleto de Fechas Futuras ---")
    # (Código sin cambios v21.1)
    start_future = last_hist_ts + pd.Timedelta(hours=1); end_future = start_future + pd.Timedelta(days=horizonte_dias, hours=23)
    df_future = pd.DataFrame(pd.date_range(start=start_future, end=end_future, freq='h', tz=TZ), columns=['ts']); df_future = df_future.iloc[:horizonte_dias * 24]
    df_future = add_time_parts(df_future)
    df_future['feriados'] = df_future['ts'].dt.date.isin(feriados_list).astype(int)
    temp_ts_series = pd.concat([df_hosting_processed['ts'].iloc[-(N_STEPS+24):], df_future['ts']])
    temp_feriados_series = pd.concat([df_hosting_processed['feriados'].iloc[-(N_STEPS+24):], df_future['feriados']])
    all_feriados_df = pd.DataFrame({'ts': temp_ts_series, 'feriados': temp_feriados_series}).drop_duplicates(subset=['ts']).set_index('ts').sort_index()
    full_range = pd.date_range(start=all_feriados_df.index.min(), end=all_feriados_df.index.max(), freq='h'); all_feriados_df = all_feriados_df.reindex(full_range).ffill()
    future_feriados_shifted_after = all_feriados_df['feriados'].shift(24).loc[df_future['ts']]
    future_feriados_shifted_before = all_feriados_df['feriados'].shift(-24).loc[df_future['ts']]
    df_future['dia_despues_feriado'] = future_feriados_shifted_after.fillna(0).astype(int)
    df_future['dia_antes_feriado'] = future_feriados_shifted_before.fillna(0).astype(int)
    print(f"  [OK] Esqueleto futuro creado: {df_future['ts'].min()} a {df_future['ts'].max()}")


    # --- 4. Pipeline Clima ---
    # (Sin cambios)
    print("\n--- Fase 4: Pipeline de Clima (Analista de Riesgos) ---")
    df_weather_future_raw = fetch_future_weather(start_future, end_future)
    df_agg_anomalies, df_per_comuna_anomalies = process_future_climate(df_weather_future_raw, baselines_clima if not baselines_clima.empty else pd.DataFrame())
    df_future = pd.merge(df_future, df_agg_anomalies, on='ts', how='left')
    numeric_cols_future = df_future.select_dtypes(include=np.number).columns
    df_future[numeric_cols_future] = df_future[numeric_cols_future].fillna(df_future[numeric_cols_future].mean()); df_future = df_future.fillna(0)
    if cols_risk and all(c in df_future.columns for c in cols_risk):
        X_risk = df_future.reindex(columns=cols_risk, fill_value=0); X_risk_s = scaler_risk.transform(X_risk)
        df_future['risk_proba'] = model_risk.predict(X_risk_s); print("  [OK] Predicciones riesgo generadas.")
    else: print("  [Adv] Faltan columnas/config risk. 'risk_proba'=0."); df_future['risk_proba'] = 0.0
    df_risk_proba_output = df_future[['ts', 'risk_proba']].copy(); alertas_json_data = generate_alerts_json(df_per_comuna_anomalies, df_risk_proba_output)

    # --- 5. Pipeline Llamadas (LSTM Base + Capping MAD) ---
    print("\n--- Fase 5: Pipeline de Llamadas (LSTM Base + Capping MAD) ---")
    df_future_features_planner = df_future.copy(); df_future_features_planner[TARGET_CALLS] = 0 # Placeholder
    X_future_df = pd.get_dummies(df_future_features_planner[hist_features_planner], columns=['month', 'semana_del_mes'])
    X_future_df = X_future_df.reindex(columns=cols_planner, fill_value=0); X_future_scaled = scaler_planner.transform(X_future_df)
    print(f"Creando secuencias inferencia Planificador ({N_STEPS} hist)...")
    X_planner_seq = create_inference_sequences(historical_context_planner, X_future_scaled, time_steps=N_STEPS)
    if X_planner_seq.shape[0] > 0:
        print(f"Prediciendo con LSTM Planner ({X_planner_seq.shape})..."); predictions_planner = model_planner.predict(X_planner_seq).flatten()
        if len(predictions_planner) == len(df_future): df_future['llamadas_hora'] = predictions_planner.clip(0) # <-- Float
        else: print(f"  [ERROR] Discrepancia longitud predicción ({len(predictions_planner)}) vs futuro ({len(df_future)})."); df_future['llamadas_hora'] = 0.0
    else: print("  [ERROR] No se crearon secuencias inferencia Planificador."); df_future['llamadas_hora'] = 0.0
    print("  [OK] Predicciones BASE llamadas (LSTM q=0.50) generadas.")

    # --- Aplicar Capping MAD ---
    df_future = aplicar_capping_mad(df_future, baselines_llamadas, k_weekday=K_MAD_WEEKDAY, k_weekend=K_MAD_WEEKEND)
    # --- Fin ---

    # --- 6. Pipeline TMO (LSTM Base) ---
    print("\n--- Fase 6: Pipeline de TMO (Analista de Operaciones LSTM Base) ---")
    # (Usa el mismo modelo q=0.50 que Planner, pero sin capping)
    if df_tmo_hist.empty: print("  [Adv] TMO_HISTORICO vacío."); last_tmo_data = pd.Series(dtype='float64')
    else: last_tmo_data = df_tmo_hist.sort_values('ts').iloc[-1]
    seed_cols = ['proporcion_comercial', 'proporcion_tecnica', 'tmo_comercial', 'tmo_tecnico']
    df_tmo_features_future = df_future.copy(); df_tmo_features_future[TARGET_CALLS] = df_tmo_features_future['llamadas_hora'] # <-- USA LLAMADAS AJUSTADAS POR CAPPING
    for col in seed_cols: df_tmo_features_future[col] = last_tmo_data.get(col, 0)
    df_tmo_features_future[TARGET_TMO] = 0 # Placeholder
    # Histórico TMO
    df_hist_tmo_merged = pd.merge(df_hosting_processed, df_tmo_hist, on='ts', how='inner')
    if not df_tmo_hist.empty:
         hist_features_tmo = ['proporcion_comercial', 'proporcion_tecnica', 'tmo_comercial', 'tmo_tecnico', TARGET_CALLS, 'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'feriados', 'dia_despues_feriado', 'dia_antes_feriado', 'es_quincena', 'es_dia_de_pago', 'month', 'semana_del_mes', 'es_domingo', 'es_madrugada', 'es_navidad', 'es_ano_nuevo', 'es_fiestas_patrias', TARGET_TMO]
         anomaly_cols_in_tmo = [c for c in cols_tmo if 'anomalia_' in c or 'pct_comunas' in c]; hist_features_tmo.extend(anomaly_cols_in_tmo)
         if 'precipitacion_x_dia_habil' in cols_tmo: hist_features_tmo.append('precipitacion_x_dia_habil')
         for f in hist_features_tmo:
              if f not in df_hist_tmo_merged.columns: df_hist_tmo_merged[f] = 0
         X_hist_tmo_df = pd.get_dummies(df_hist_tmo_merged[hist_features_tmo], columns=['month', 'semana_del_mes'])
         X_hist_tmo_df = X_hist_tmo_df.reindex(columns=cols_tmo, fill_value=0); X_hist_tmo_scaled = scaler_tmo.transform(X_hist_tmo_df)
         if len(X_hist_tmo_scaled) < N_STEPS: print(f"  [Adv] No hist TMO ({len(X_hist_tmo_scaled)}) para contexto ({N_STEPS})."); historical_context_tmo = np.zeros((N_STEPS, len(cols_tmo)))
         else: historical_context_tmo = X_hist_tmo_scaled[-N_STEPS:]
    else: print("  [Adv] Histórico TMO vacío."); historical_context_tmo = np.zeros((N_STEPS, len(cols_tmo)))
    # Futuro TMO
    X_future_tmo_df = pd.get_dummies(df_tmo_features_future[hist_features_tmo], columns=['month', 'semana_del_mes'])
    X_future_tmo_df = X_future_tmo_df.reindex(columns=cols_tmo, fill_value=0); X_future_tmo_scaled = scaler_tmo.transform(X_future_tmo_df)
    print(f"Creando secuencias inferencia TMO ({N_STEPS} hist)...")
    X_tmo_seq = create_inference_sequences(historical_context_tmo, X_future_tmo_scaled, time_steps=N_STEPS)
    if X_tmo_seq.shape[0] > 0:
        print(f"Prediciendo con LSTM TMO ({X_tmo_seq.shape})..."); predictions_tmo = model_tmo.predict(X_tmo_seq).flatten()
        if len(predictions_tmo) == len(df_future): df_future['tmo_hora'] = predictions_tmo.clip(0); print("  [OK] Predicciones TMO (LSTM q=0.50) generadas.")
        else: print(f"  [ERROR] Discrepancia longitud TMO ({len(predictions_tmo)}) vs futuro ({len(df_future)})."); df_future['tmo_hora'] = 0.0
    else: print("  [ERROR] No se crearon secuencias inferencia TMO."); df_future['tmo_hora'] = 0.0

    # --- 7. Generar Salidas Finales ---
    print("\n--- Fase 7: Generando Archivos JSON de Salida ---")
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    df_future['agentes_requeridos'] = calculate_erlang_agents(df_future['llamadas_hora'], df_future['tmo_hora'])
    df_horaria = df_future[['ts', 'llamadas_hora', 'tmo_hora', 'agentes_requeridos']].copy()
    df_horaria['ts'] = df_horaria['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    output_path_horaria = os.path.join(PUBLIC_DIR, "prediccion_horaria.json")
    df_horaria.to_json(output_path_horaria, orient='records', indent=2, force_ascii=False)
    print(f"  [OK] Archivo 'prediccion_horaria.json' guardado.")

    df_horaria_para_diaria = df_future.copy(); df_horaria_para_diaria['fecha'] = df_horaria_para_diaria['ts'].dt.date
    df_horaria_para_diaria['tmo_ponderado_num'] = df_horaria_para_diaria['tmo_hora'] * df_horaria_para_diaria['llamadas_hora']
    df_diaria_agg = df_horaria_para_diaria.groupby('fecha').agg(llamadas_totales_dia=('llamadas_hora', 'sum'), tmo_ponderado_num=('tmo_ponderado_num', 'sum'))
    df_diaria_agg['tmo_promedio_diario'] = df_diaria_agg['tmo_ponderado_num'] / (df_diaria_agg['llamadas_totales_dia'] + 1e-6)
    if (df_diaria_agg['llamadas_totales_dia'] == 0).any():
        tmo_simple_avg = df_horaria_para_diaria.groupby('fecha')['tmo_hora'].mean().fillna(0) # Rellenar NaN
        df_diaria_agg['tmo_promedio_diario'] = df_diaria_agg['tmo_promedio_diario'].where(df_diaria_agg['llamadas_totales_dia'] > 0, tmo_simple_avg).fillna(0) # Rellenar NaN
    df_diaria_agg = df_diaria_agg.reset_index()[['fecha', 'llamadas_totales_dia', 'tmo_promedio_diario']]
    df_diaria_agg['fecha'] = df_diaria_agg['fecha'].astype(str); df_diaria_agg['llamadas_totales_dia'] = df_diaria_agg['llamadas_totales_dia'].astype(int)
    output_path_diaria = os.path.join(PUBLIC_DIR, "Predicion_daria.json")
    df_diaria_agg.to_json(output_path_diaria, orient='records', indent=2, force_ascii=False)
    print(f"  [OK] Archivo 'Predicion_daria.json' guardado.")

    output_path_alertas = os.path.join(PUBLIC_DIR, "alertas_climaticas.json")
    with open(output_path_alertas, 'w', encoding='utf-8') as f: json.dump(alertas_json_data, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Archivo 'alertas_climaticas.json' guardado.")

    print("\n" + "="*60); print("PIPELINE DE INFERENCIA COMPLETADO EXITOSAMENTE."); print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Inferencia IA para Predicción de Tráfico.")
    parser.add_argument("--horizonte", type=int, default=120, help="Horizonte predicción días")
    args = parser.parse_args()
    main(horizonte_dias=args.horizonte)
