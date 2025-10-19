import os
import argparse
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# --- CONFIGURACIÓN GLOBAL ---
TZ = 'America/Santiago'; os.environ['TZ'] = TZ
BASE_DIR = os.path.dirname(os.path.abspath(__file__)); ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data"); MODEL_DIR = os.path.join(ROOT_DIR, "models"); PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
# Archivos (nombres estándar v7)
PLANNER_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_planner.keras"); PLANNER_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_planner.pkl"); PLANNER_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_planner.json")
RISK_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_riesgos.keras"); RISK_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_riesgos.pkl"); RISK_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_riesgos.json"); RISK_BASELINES_FILE = os.path.join(MODEL_DIR, "baselines_clima.pkl")
TMO_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_tmo.keras"); TMO_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_tmo.pkl"); TMO_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_tmo.json")
HOSTING_FILE = os.path.join(DATA_DIR, "historical_data.csv"); TMO_FILE = os.path.join(DATA_DIR, "TMO_HISTORICO.csv"); FERIADOS_FILE = os.path.join(DATA_DIR, "Feriados_Chilev2.csv"); CLIMA_HIST_FILE = os.path.join(DATA_DIR, "historical_data.csv")
TARGET_CALLS = "recibidos_nacional"; TARGET_TMO = "tmo_general"

# --- AJUSTE v27: Parámetros de Post-Procesamiento (de forecast3m.py) ---
MAD_K = 5.0  # K base (lunes-viernes)
MAD_K_WEEKEND = 6.5 # K fin de semana
# --- Fin Ajuste ---

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'; warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl'); warnings.filterwarnings('ignore', category=FutureWarning)
# Suprimir advertencias de Pandas sobre chained assignment (común en post-proceso)
pd.options.mode.chained_assignment = None # default='warn'

# --- FUNCIONES DE UTILIDAD (main.py + forecast3m.py) ---

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

# add_time_parts (v7) - Usada por el modelo MLP
def add_time_parts(df):
    df_copy = df.copy()
    df_copy["dow"] = df_copy["ts"].dt.dayofweek
    df_copy["month"] = df_copy["ts"].dt.month
    df_copy["hour"] = df_copy["ts"].dt.hour
    df_copy["day"] = df_copy["ts"].dt.day
    df_copy["es_dia_de_pago"] = df_copy['day'].isin([1, 2, 15, 16, 29, 30, 31]).astype(int)
    df_copy["sin_hour"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
    df_copy["cos_hour"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
    df_copy["sin_dow"] = np.sin(2 * np.pi * df_copy["dow"] / 7)
    df_copy["cos_dow"] = np.cos(2 * np.pi * df_copy["dow"] / 7)
    return df_copy

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

# --- AJUSTE v27: INICIO FUNCIONES DE POST-PROCESAMIENTO (de forecast3m.py) ---
# Adaptadas para usar un DataFrame con columna 'ts' en lugar de DatetimeIndex

def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if num is not None and den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
    return num / den

def load_holidays(csv_path, tz=TZ):
    """Carga feriados y devuelve un set de objetos date."""
    if not os.path.exists(csv_path):
        print(f"  [Adv] No se encontró archivo de feriados en {csv_path}. No se aplicarán ajustes.")
        return set()
    try:
        fer = pd.read_csv(csv_path)
        if "Fecha" not in fer.columns:
             # Probar con 'fecha' minúscula
             if "fecha" in fer.columns:
                 fer.rename(columns={"fecha":"Fecha"}, inplace=True)
             else:
                 print("  [Adv] CSV de feriados no tiene columna 'Fecha'."); return set()
        # Intentar formato dd-mm-yyyy primero
        try:
            fechas = pd.to_datetime(fer["Fecha"].astype(str), format='%d-%m-%Y', errors='coerce').dt.date
        except Exception:
            fechas = pd.to_datetime(fer["Fecha"].astype(str), errors='coerce').dt.date
        
        return set(fechas.dropna())
    except Exception as e:
        print(f"  [Error] No se pudo cargar feriados: {e}"); return set()

def mark_holidays_series(ts_series, holidays_set):
    """Crea una Serie booleana 'is_holiday' desde una Serie de timestamps."""
    if holidays_set is None or not holidays_set:
        return pd.Series(False, index=ts_series.index, name="is_holiday")
    # Asegurar que 'ts_series' esté en la zona horaria correcta antes de sacar .date
    if ts_series.dt.tz is None:
        ts_dates = ts_series.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").dt.date
    else:
        ts_dates = ts_series.dt.tz_convert(TZ).dt.date
    return ts_dates.isin(holidays_set).astype(bool).rename("is_holiday")

def compute_seasonal_weights(df_hist, col, weeks=8, clip_min=0.75, clip_max=1.30):
    """w(dow,h) = mediana_hist(dow,h) / mediana_hist_por_hora(h)"""
    print(f"  Calculando recalibración estacional (últimas {weeks} semanas)...")
    d = df_hist.copy()
    if len(d) == 0: return { (dow,h): 1.0 for dow in range(7) for h in range(24) }
    end = d['ts'].max(); start = end - pd.Timedelta(weeks=weeks)
    d = d.loc[d['ts'] >= start]
    if 'dow' not in d.columns: d = add_time_parts(d) # Asegurar dow/hour
    
    med_dow_hour = d.groupby(["dow","hour"])[col].median()
    med_hour = d.groupby("hour")[col].median()
    weights = {}
    for dow in range(7):
        for h in range(24):
            num = med_dow_hour.get((dow,h), np.nan); den = med_hour.get(h, np.nan)
            w = _safe_ratio(num, den, fallback=1.0)
            weights[(dow,h)] = float(np.clip(w, clip_min, clip_max))
    return weights

def apply_seasonal_weights(df_future, weights, col_name="pred_llamadas"):
    """Aplica pesos estacionales a un dataframe futuro."""
    df = df_future.copy()
    if 'dow' not in df.columns: df = add_time_parts(df) # Asegurar dow/hour
    idx = list(zip(df["dow"].values, df["hour"].values))
    w = np.array([weights.get(key, 1.0) for key in idx], dtype=float)
    df[col_name] = (df[col_name].astype(float) * w)
    return df

def baseline_from_history(df_hist, col):
    """Calcula mediana, MAD y q95 por dow y hour del histórico."""
    print(f"  Calculando baselines robustos (MAD, q95) para '{col}'...")
    if df_hist.empty or col not in df_hist.columns:
        print("  [Advertencia] Histórico vacío o sin target_col. No se calcularán baselines MAD.")
        return pd.DataFrame()
    d = add_time_parts(df_hist[['ts', col]].copy())
    g = d.groupby(["dow", "hour"])[col]
    base = g.median().rename("med").to_frame()
    median_map = base['med'].to_dict()
    def mad_calc(x):
        med = median_map.get((x.name[0], x.name[1]), np.nan)
        if pd.isna(med): return np.nan
        return np.median(np.abs(x - med))
    base['mad'] = g.apply(mad_calc)
    base['q95'] = g.quantile(0.95)
    
    # Rellenar NaNs
    global_mad_median = base['mad'].median(); global_q95_median = base['q95'].median()
    base['mad'].fillna(global_mad_median if not pd.isna(global_mad_median) else 1.0, inplace=True)
    base['q95'].fillna(global_q95_median if not pd.isna(global_q95_median) else base['med'], inplace=True)
    base['mediana'].fillna(d[col].median(), inplace=True)
    base['mad'] = np.maximum(base['mad'], 1e-6) # Evitar MAD=0
    return base.reset_index()

def apply_peak_smoothing_history(df_future, col, base, k_weekday=K_MAD_WEEKDAY, k_weekend=K_MAD_WEEKEND):
    """Aplica capping (recorte) a picos basado en baselines históricos."""
    df = df_future.copy()
    if base.empty: print("    [Advertencia] Baselines vacíos. Omitiendo capping MAD."); return df
    if 'dow' not in df.columns: df = add_time_parts(df)
    
    df_merged = pd.merge(df, base, on=['dow', 'hour'], how='left')
    # Rellenar NaNs en baselines (si alguna combinación dow/hour no estaba en histórico)
    df_merged['mediana'].fillna(base['mediana'].mean(), inplace=True)
    df_merged['mad'].fillna(base['mad'].mean(), inplace=True)
    df_merged['q95'].fillna(base['q95'].mean(), inplace=True)
    
    K = np.where(df_merged["dow"].isin([5, 6]), k_weekend, k_weekday).astype(float) # 5=sáb,6=dom
    upper_cap = df_merged["mediana"].values + K * df_merged["mad"].values
    
    # Recortar picos que superan AMBOS, el cap dinámico (MAD) Y el q95 histórico
    mask = (df_merged[col].astype(float).values > upper_cap) & (df_merged[col].astype(float).values > df_merged["q95"].values)
    n_capped = mask.sum()
    if n_capped > 0:
        print(f"    - Recortando {n_capped} picos (Capping MAD) a límite histórico.")
        df_merged.loc[mask, col] = upper_cap[mask]
    
    return df_merged[df.columns] # Devolver solo las columnas originales

def compute_holiday_factors(df_hist, holidays_set, col_calls=TARGET_CALLS, col_tmo=TARGET_TMO):
    """Calcula factores multiplicativos por hora para feriados."""
    print("  Calculando factores de ajuste por feriados...")
    dfh = add_time_parts(df_hist[[col_calls, col_tmo, 'ts']].copy())
    dfh["is_holiday"] = mark_holidays_series(dfh['ts'], holidays_set).values
    
    # Calcular medianas
    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_hol_tmo   = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median()
    med_nor_tmo   = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()
    
    # Fallback global
    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median(); g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    g_hol_tmo   = dfh[dfh["is_holiday"]][col_tmo].median(); g_nor_tmo   = dfh[~dfh["is_holiday"]][col_tmo].median()
    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75) # Asume 75% si no hay datos
    global_tmo_factor   = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00) # Asume 100% si no hay datos

    factors_calls_by_hour = {h: _safe_ratio(med_hol_calls.get(h, np.nan), med_nor_calls.get(h, np.nan), fallback=global_calls_factor) for h in range(24)}
    factors_tmo_by_hour   = {h: _safe_ratio(med_hol_tmo.get(h, np.nan),   med_nor_tmo.get(h, np.nan),   fallback=global_tmo_factor)   for h in range(24)}

    # Aplicar clips (límites) a los factores
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.20)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}
    return factors_calls_by_hour, factors_tmo_by_hour, global_calls_factor, global_tmo_factor

def apply_holiday_adjustment(df_future, holidays_set, factors_calls_by_hour, factors_tmo_by_hour, col_calls="pred_llamadas", col_tmo="pred_tmo_seg"):
    """Aplica factores de feriado a predicciones futuras."""
    df = add_time_parts(df_future.copy())
    is_hol = mark_holidays_series(df['ts'], holidays_set).values
    if is_hol.sum() == 0:
        print("    - No hay feriados en el horizonte futuro. Omitiendo ajuste.")
        return df_future # Devolver original si no hay feriados
    
    hours = df["hour"].values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])
    
    # Aplicar solo a las filas que son feriado
    df.loc[is_hol, col_calls] = (df.loc[is_hol, col_calls].astype(float) * call_f[is_hol])
    df.loc[is_hol, col_tmo]  = (df.loc[is_hol, col_tmo].astype(float)  * tmo_f[is_hol])
    print(f"    - Ajuste de feriados aplicado a {is_hol.sum()} horas.")
    return df[df_future.columns] # Devolver solo columnas originales
# --- AJUSTE v27: FIN FUNCIONES DE POST-PROCESAMIENTO ---


# --- FUNCIONES DEL PIPELINE DE INFERENCIA ---
# (fetch_future_weather, process_future_climate, generate_alerts_json - sin cambios)
def fetch_future_weather(start_date, end_date):
    print("    [Clima] SIMULANDO API de clima futuro...");
    try: df_hist = read_data(CLIMA_HIST_FILE)
    except FileNotFoundError: print(f"    [Clima] ADVERTENCIA: {CLIMA_HIST_FILE} no encontrado. Dummy."); comunas = ['Santiago']; dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts'])); df_simulado['temperatura'] = 15; df_simulado['precipitacion'] = 0; df_simulado['lluvia'] = 0; return df_simulado.reset_index()
    df_hist = ensure_ts_and_tz(df_hist); df_hist = normalize_climate_columns(df_hist); climate_cols_found = [col for col in ['temperatura', 'precipitacion', 'lluvia'] if col in df_hist.columns]
    if not climate_cols_found: print(f"    [Clima] ADVERTENCIA: No cols clima en {CLIMA_HIST_FILE}. Dummy."); comunas = ['Santiago']; dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts'])); df_simulado['temperatura'] = 15; df_simulado['precipitacion'] = 0; df_simulado['lluvia'] = 0; return df_simulado.reset_index()
    if 'comuna' not in df_hist.columns: print("    [Clima] ADVERTENCIA: 'comuna' no encontrada. Dummy 'Santiago'."); df_hist['comuna'] = 'Santiago'
    future_dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ); df_future_list = []
    for date in future_dates:
        try: sim_date = date.replace(year=date.year - 1); data_sim = df_hist[df_hist['ts'] == sim_date];
        except ValueError: sim_date = date - pd.Timedelta(days=365); data_sim = df_hist[df_hist['ts'] == sim_date]
        if not data_sim.empty: data_sim = data_sim.copy(); data_sim['ts'] = date; df_future_list.append(data_sim) # Use .copy()
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
    print("="*60); print(f"INICIANDO PIPELINE DE INFERENCIA (v_main 27 - MLP Base + Post-Proceso)"); print(f"Zona Horaria: {TZ} | Horizonte: {horizonte_dias} días"); print("="*60) # v27

    # --- 1. Cargar Modelos y Artefactos (v7) ---
    print("\n--- Fase 1: Cargando Modelos y Artefactos (v7) ---")
    try:
        model_planner = tf.keras.models.load_model(PLANNER_MODEL_FILE); scaler_planner = joblib.load(PLANNER_SCALER_FILE)
        with open(PLANNER_COLS_FILE, 'r') as f: cols_planner = json.load(f)
        model_risk = tf.keras.models.load_model(RISK_MODEL_FILE); scaler_risk = joblib.load(RISK_SCALER_FILE)
        try: with open(RISK_COLS_FILE, 'r') as f: cols_risk = json.load(f)
        except FileNotFoundError: print(f"  [Adv] {RISK_COLS_FILE} no encontrado."); cols_risk = []
        try: baselines_clima = joblib.load(RISK_BASELINES_FILE)
        except FileNotFoundError: print(f"  [Adv] {RISK_BASELINES_FILE} no encontrado."); baselines_clima = pd.DataFrame()
        model_tmo = tf.keras.models.load_model(TMO_MODEL_FILE); scaler_tmo = joblib.load(TMO_SCALER_FILE)
        with open(TMO_COLS_FILE, 'r') as f: cols_tmo = json.load(f)
        print("  [OK] Todos los modelos v7 cargados.")
    except Exception as e: print(f"  [ERROR] Falla crítica al cargar artefactos: {e}"); print("  Asegúrate que archivos v7 existan en 'models/'."); return

    # --- 2. Cargar Datos Históricos ---
    print("\n--- Fase 2: Cargando Datos Históricos ---")
    # --- AJUSTE v27: Usar nueva función load_holidays ---
    holidays_set = load_holidays(FERIADOS_FILE, tz=TZ)
    # --- Fin Ajuste ---
    df_hosting_full = read_data(HOSTING_FILE); df_hosting = ensure_ts_and_tz(df_hosting_full)
    if 'feriados' not in df_hosting.columns: print("  [Info] Creando columna 'feriados'."); df_hosting['feriados'] = mark_holidays_series(df_hosting['ts'], holidays_set).values
    else: df_hosting['feriados'] = pd.to_numeric(df_hosting['feriados'], errors='coerce').fillna(0).astype(int)
    if 'recibidos' in df_hosting.columns and TARGET_CALLS not in df_hosting.columns: df_hosting = df_hosting.rename(columns={'recibidos': TARGET_CALLS})
    elif TARGET_CALLS not in df_hosting.columns: raise ValueError(f"No se encontró {TARGET_CALLS} ni 'recibidos'.")
    df_hosting_agg = df_hosting.groupby("ts").agg({TARGET_CALLS: 'sum', 'feriados': 'max'}).reset_index()
    
    df_tmo_hist = read_data(TMO_FILE); df_tmo_hist = ensure_ts_and_tz(df_tmo_hist); df_tmo_hist.columns = [c.lower().strip().replace(' ', '_') for c in df_tmo_hist.columns]; df_tmo_hist = df_tmo_hist.rename(columns={'tmo_general': TARGET_TMO})
    if TARGET_TMO not in df_tmo_hist.columns and all(c in df_tmo_hist.columns for c in ['tmo_comercial', 'q_comercial', 'tmo_tecnico', 'q_tecnico', 'q_general']): df_tmo_hist[TARGET_TMO] = (df_tmo_hist['tmo_comercial'] * df_tmo_hist['q_comercial'] + df_tmo_hist['tmo_tecnico'] * df_tmo_hist['q_tecnico']) / (df_tmo_hist['q_general'] + 1e-6)
    if 'q_llamadas_comercial' in df_tmo_hist.columns and 'q_llamadas_general' in df_tmo_hist.columns: df_tmo_hist['proporcion_comercial'] = df_tmo_hist['q_llamadas_comercial'] / (df_tmo_hist['q_llamadas_general'] + 1e-6); df_tmo_hist['proporcion_tecnica'] = df_tmo_hist['q_llamadas_tecnico'] / (df_tmo_hist['q_llamadas_general'] + 1e-6)
    else: print("  [Adv] No cols q_llamadas TMO."); df_tmo_hist['proporcion_comercial'] = 0; df_tmo_hist['proporcion_tecnica'] = 0
    
    # --- AJUSTE v27: Mergear TMO para tener historial completo para post-proceso ---
    df_hist_merged = pd.merge(df_hosting_agg, df_tmo_hist[['ts', TARGET_TMO]], on='ts', how='left')
    # Rellenar TMOs faltantes (ej. ffill/bfill) para tener data completa
    df_hist_merged[TARGET_TMO].fillna(method='ffill', inplace=True)
    df_hist_merged[TARGET_TMO].fillna(method='bfill', inplace=True)
    df_hist_merged[TARGET_TMO].fillna(df_hist_merged[TARGET_TMO].mean(), inplace=True) # Relleno final
    df_hist_merged.dropna(subset=[TARGET_CALLS, TARGET_TMO], inplace=True) # Quitar si algo quedó nulo
    df_hosting_processed = add_time_parts(df_hist_merged)
    # --- Fin Ajuste ---

    last_hist_ts = df_hosting_processed['ts'].max(); print(f"  [OK] Datos históricos cargados. Último timestamp: {last_hist_ts}")

    # --- 3. Generar Esqueleto Futuro ---
    print("\n--- Fase 3: Generando Esqueleto de Fechas Futuras ---")
    start_future = last_hist_ts + pd.Timedelta(hours=1); end_future = start_future + pd.Timedelta(days=horizonte_dias, hours=23)
    df_future = pd.DataFrame(pd.date_range(start=start_future, end=end_future, freq='h', tz=TZ), columns=['ts']); df_future = df_future.iloc[:horizonte_dias * 24]
    df_future = add_time_parts(df_future) # add_time_parts v7
    df_future['feriados'] = mark_holidays_series(df_future['ts'], holidays_set).values.astype(int)
    print(f"  [OK] Esqueleto futuro creado: {df_future['ts'].min()} a {df_future['ts'].max()}")

    # --- 4. Pipeline Clima ---
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

    # --- 5. Pipeline Llamadas (MLP v7 Base) ---
    print("\n--- Fase 5: Pipeline de Llamadas (Planificador MLP v7 Base) ---")
    df_full = pd.concat([df_hosting_processed, df_future], ignore_index=True).sort_values('ts')
    for lag in [24, 48, 72, 168]: df_full[f'lag_{lag}'] = df_full[TARGET_CALLS].shift(lag)
    for window in [24, 72, 168]: df_full[f'ma_{window}'] = df_full[TARGET_CALLS].shift(1).rolling(window, min_periods=1).mean()
    df_future_features = df_full[df_full['ts'] >= start_future].copy()
    X_planner = pd.get_dummies(df_future_features, columns=['dow', 'month', 'hour']); X_planner = X_planner.reindex(columns=cols_planner, fill_value=0)
    numeric_cols_planner = X_planner.select_dtypes(include=np.number).columns
    means = X_planner[numeric_cols_planner].mean(); X_planner[numeric_cols_planner] = X_planner[numeric_cols_planner].fillna(means).fillna(0)
    X_planner_s = scaler_planner.transform(X_planner)
    df_future['llamadas_hora'] = model_planner.predict(X_planner_s).clip(0) # <-- Dejar como float
    print("  [OK] Predicciones BASE llamadas (MLP v7) generadas.")

    # --- 6. Pipeline TMO (MLP v7 Base) ---
    print("\n--- Fase 6: Pipeline de TMO (Analista de Operaciones MLP v7 Base) ---")
    if df_tmo_hist.empty: print("  [Adv] TMO_HISTORICO vacío."); last_tmo_data = pd.Series(dtype='float64')
    else: last_tmo_data = df_tmo_hist.sort_values('ts').iloc[-1]
    seed_cols = ['proporcion_comercial', 'proporcion_tecnica', 'tmo_comercial', 'tmo_tecnico']
    df_tmo_features_future = df_future.copy(); df_tmo_features_future[TARGET_CALLS] = df_tmo_features_future['llamadas_hora'] # Usa llamadas base
    for col in seed_cols: df_tmo_features_future[col] = last_tmo_data.get(col, 0)
    X_tmo = pd.get_dummies(df_tmo_features_future, columns=['dow', 'month', 'hour']); X_tmo = X_tmo.reindex(columns=cols_tmo, fill_value=0)
    numeric_cols_tmo = X_tmo.select_dtypes(include=np.number).columns
    means_tmo = X_tmo[numeric_cols_tmo].mean(); X_tmo[numeric_cols_tmo] = X_tmo[numeric_cols_tmo].fillna(means_tmo).fillna(0)
    X_tmo_s = scaler_tmo.transform(X_tmo)
    df_future['tmo_hora'] = model_tmo.predict(X_tmo_s).clip(0) # <-- Dejar como float
    print("  [OK] Predicciones BASE TMO (MLP v7) generadas.")

    # --- AJUSTE v27: Nueva Fase de Post-Procesamiento ---
    print("\n--- Fase 6.5: Aplicando Post-Procesamiento (Lógica forecast3m.py) ---")
    # Renombrar columnas para que coincidan con las funciones de forecast3m.py
    df_future_post = df_future.rename(columns={'llamadas_hora': 'pred_llamadas', 'tmo_hora': 'pred_tmo_seg'})

    # 1. Recalibración Estacional
    seasonal_w = compute_seasonal_weights(df_hosting_processed, TARGET_CALLS, weeks=8)
    df_future_post = apply_seasonal_weights(df_future_post, seasonal_w, col_name="pred_llamadas")

    # 2. Suavizado/Capping MAD
    base_hist = baseline_from_history(df_hosting_processed, TARGET_CALLS)
    df_future_post = apply_peak_smoothing_history(df_future_post, 'pred_llamadas', base_hist, k_weekday=K_MAD_WEEKDAY, k_weekend=K_MAD_WEEKEND)

    # 3. Ajuste de Feriados (para llamadas y TMO)
    f_calls, f_tmo, g_calls, g_tmo = compute_holiday_factors(df_hosting_processed, holidays_set, TARGET_CALLS, TARGET_TMO)
    print(f"  Factor global feriado (info): llamadas={g_calls:.3f}, TMO={g_tmo:.3f}")
    df_future_post = apply_holiday_adjustment(df_future_post, holidays_set, f_calls, f_tmo, col_calls="pred_llamadas", col_tmo="pred_tmo_seg")
    
    # Devolver nombres y tipos de datos correctos
    df_future['llamadas_hora'] = df_future_post['pred_llamadas'].round().clip(0).astype(int)
    df_future['tmo_hora'] = df_future_post['pred_tmo_seg'].clip(0)
    print("--- Post-Procesamiento Completado ---")
    # --- Fin Ajuste ---

    # --- 7. Generar Salidas Finales ---
    print("\n--- Fase 7: Generando Archivos JSON de Salida (Ajustados) ---")
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
        tmo_simple_avg = df_horaria_para_diaria.groupby('fecha')['tmo_hora'].mean().fillna(0)
        df_diaria_agg['tmo_promedio_diario'] = df_diaria_agg['tmo_promedio_diario'].where(df_diaria_agg['llamadas_totales_dia'] > 0, tmo_simple_avg).fillna(0)
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
