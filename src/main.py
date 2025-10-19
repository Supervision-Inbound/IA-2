import os
import argparse
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K # Para Quantile Loss

# --- CONFIGURACIÓN GLOBAL ---
TZ = 'America/Santiago'
os.environ['TZ'] = TZ

# Definición de rutas relativas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

# Nombres de archivos de artefactos (v14)
PLANNER_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_planner.keras")
PLANNER_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_planner.pkl")
PLANNER_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_planner.json")

RISK_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_riesgos.keras")
RISK_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_riesgos.pkl")
RISK_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_riesgos.json")
RISK_BASELINES_FILE = os.path.join(MODEL_DIR, "baselines_clima.pkl")

TMO_MODEL_FILE = os.path.join(MODEL_DIR, "modelo_tmo.keras")
TMO_SCALER_FILE = os.path.join(MODEL_DIR, "scaler_tmo.pkl")
TMO_COLS_FILE = os.path.join(MODEL_DIR, "training_columns_tmo.json")

# Nombres de archivos de datos
HOSTING_FILE = os.path.join(DATA_DIR, "historical_data.csv")
TMO_FILE = os.path.join(DATA_DIR, "TMO_HISTORICO.csv")
FERIADOS_FILE = os.path.join(DATA_DIR, "Feriados_Chilev2.csv")
CLIMA_HIST_FILE = os.path.join(DATA_DIR, "historical_data.csv") # Se usa para simular clima

TARGET_CALLS = "recibidos_nacional"
TARGET_TMO = "tmo_general"

# --- Cuantil usado en Entrenamiento (v14) ---
QUANTILE_P = 0.65

# Suprimir advertencias
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
warnings.filterwarnings('ignore', category=FutureWarning) # Suprimir warnings de Pandas/TF

# --- Función Quantile Loss (Necesaria para cargar modelos) ---
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
    return loss
# --- Fin ---


# --- FUNCIONES DE UTILIDAD ---

def read_data(path, hoja=None):
    path_lower = path.lower()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo de datos no encontrado: {path}")
    if path_lower.endswith(".csv"):
        try:
            df = pd.read_csv(path, low_memory=False)
            if df.shape[1] == 1 and df.iloc[0,0] is not None and ';' in str(df.iloc[0,0]):
                df = pd.read_csv(path, delimiter=';', low_memory=False)
            return df
        except Exception:
            try:
                return pd.read_csv(path, delimiter=';', low_memory=False)
            except Exception as e2:
                 raise ValueError(f"No se pudo leer CSV {path} con ',' ni ';': {e2}")
    elif path_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=hoja if hoja is not None else 0)
    else:
        raise ValueError(f"Formato no soportado: {path}")

def ensure_ts_and_tz(df):
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    date_col = next((c for c in df.columns if 'fecha' in c), None)
    hour_col = next((c for c in df.columns if 'hora' in c), None)

    if 'ts' not in df.columns:
        if not date_col or not hour_col:
            raise ValueError("No se encontraron 'fecha' y 'hora' o 'ts' en el DataFrame.")
        try:
             df["ts"] = pd.to_datetime(df[date_col] + ' ' + df[hour_col], format='%d-%m-%Y %H:%M:%S', errors='raise')
        except (ValueError, TypeError):
             print(f"  [Adv] Formato dd-mm-yyyy no detectado para fecha/hora. Intentando inferir.")
             df["ts"] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[hour_col].astype(str), errors='coerce')
    else:
        df["ts"] = pd.to_datetime(df["ts"], errors='coerce')

    df = df.dropna(subset=["ts"])

    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    else:
        df["ts"] = df["ts"].dt.tz_convert(TZ)

    df = df.dropna(subset=["ts"])
    return df.sort_values("ts")

def add_time_parts(df):
    df_copy = df.copy()
    df_copy["dow"] = df_copy["ts"].dt.dayofweek
    df_copy["month"] = df_copy["ts"].dt.month
    df_copy["hour"] = df_copy["ts"].dt.hour
    df_copy["day"] = df_copy["ts"].dt.day
    df_copy["semana_del_mes"] = (df_copy["day"] - 1) // 7 + 1

    # --- Features v14 ---
    df_copy["es_domingo"] = (df_copy["dow"] == 6).astype(int)
    df_copy["es_madrugada"] = (df_copy["hour"] < 6).astype(int)
    df_copy["es_navidad"] = ((df_copy["month"] == 12) & (df_copy["day"] == 25)).astype(int)
    df_copy["es_ano_nuevo"] = ((df_copy["month"] == 1) & (df_copy["day"] == 1)).astype(int)
    # Dia despues feriado se calcula después de cargar feriados
    # --- Fin ---

    df_copy["sin_hour"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
    df_copy["cos_hour"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
    df_copy["sin_dow"] = np.sin(2 * np.pi * df_copy["dow"] / 7)
    df_copy["cos_dow"] = np.cos(2 * np.pi * df_copy["dow"] / 7)
    return df_copy

def normalize_climate_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        'temperatura': ['temperature_2m', 'temperatura', 'temp', 'temp_2m'],
        'precipitacion': ['precipitation', 'precipitacion', 'precipitación', 'rain_mm', 'rain'],
        'lluvia': ['rain', 'lluvia', 'rainfall']
    }
    df_renamed = df.copy()
    df_renamed.columns = [c.lower().strip().replace(' ', '_') for c in df_renamed.columns]
    for standard_name, possible_names in column_map.items():
        for name in possible_names:
            if name in df_renamed.columns:
                df_renamed.rename(columns={name: standard_name}, inplace=True)
                break
    return df_renamed

def calculate_erlang_agents(calls_per_hour, tmo_seconds, occupancy_target=0.85):
    if calls_per_hour.sum() == 0:
        return pd.Series(0, index=calls_per_hour.index)
    traffic_intensity = (calls_per_hour * tmo_seconds) / 3600
    agents = np.ceil(traffic_intensity / occupancy_target)
    agents[traffic_intensity > 0] = agents[traffic_intensity > 0].apply(lambda x: max(x, 1))
    return agents.astype(int)

# --- FUNCIONES DEL PIPELINE DE INFERENCIA ---

def fetch_future_weather(start_date, end_date):
    print("    [Clima] SIMULANDO API de clima futuro...")
    try:
        df_hist = read_data(CLIMA_HIST_FILE)
    except FileNotFoundError:
        print(f"    [Clima] ADVERTENCIA: No se encontró {CLIMA_HIST_FILE}. Generando datos dummy.")
        # Generar datos dummy
        comunas = ['Santiago', 'Puente Alto', 'Maipu'] # Ejemplo
        try:
            df_comunas_coords = read_data(os.path.join(DATA_DIR, "Comunas_Cordenadas.csv"))
            if 'comuna' in df_comunas_coords.columns:
                comunas = df_comunas_coords['comuna'].unique().tolist()
        except Exception:
            print("    [Clima] Adv: No se encontró Comunas_Cordenadas.csv, usando comunas dummy.")

        dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ)
        df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts']))
        df_simulado['temperatura'] = np.random.uniform(10, 25, size=len(df_simulado))
        df_simulado['precipitacion'] = np.random.uniform(0, 1, size=len(df_simulado))
        df_simulado['lluvia'] = np.random.uniform(0, 1, size=len(df_simulado))
        return df_simulado.reset_index()

    df_hist = ensure_ts_and_tz(df_hist)
    df_hist = normalize_climate_columns(df_hist)
    climate_cols_found = [col for col in ['temperatura', 'precipitacion', 'lluvia'] if col in df_hist.columns]

    if not climate_cols_found:
        print(f"    [Clima] ADVERTENCIA: No se encontraron columnas de clima en {CLIMA_HIST_FILE}. Generando datos dummy.")
        # Reutilizar el bloque dummy de arriba
        comunas = ['Santiago', 'Puente Alto', 'Maipu']
        try:
            df_comunas_coords = read_data(os.path.join(DATA_DIR, "Comunas_Cordenadas.csv"))
            if 'comuna' in df_comunas_coords.columns:
                comunas = df_comunas_coords['comuna'].unique().tolist()
        except Exception:
             print("    [Clima] Adv: No se encontró Comunas_Cordenadas.csv, usando comunas dummy.")
        dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ)
        df_simulado = pd.DataFrame(index=pd.MultiIndex.from_product([comunas, dates], names=['comuna', 'ts']))
        df_simulado['temperatura'] = np.random.uniform(10, 25, size=len(df_simulado))
        df_simulado['precipitacion'] = np.random.uniform(0, 1, size=len(df_simulado))
        df_simulado['lluvia'] = np.random.uniform(0, 1, size=len(df_simulado))
        return df_simulado.reset_index()


    if 'comuna' not in df_hist.columns:
        print("    [Clima] ADVERTENCIA: 'comuna' no encontrada en archivo de clima. Usando 'Santiago' como dummy.")
        df_hist['comuna'] = 'Santiago'

    # Simulación
    future_dates = pd.date_range(start=start_date, end=end_date, freq='h', tz=TZ)
    df_future_list = []
    for date in future_dates:
        try:
            sim_date = date.replace(year=date.year - 1)
            data_sim = df_hist[df_hist['ts'] == sim_date]
            if not data_sim.empty:
                data_sim['ts'] = date
                df_future_list.append(data_sim)
        except Exception:
            continue

    if not df_future_list:
        print("    [Clima] No se pudo simular (año anterior). Usando última semana.")
        last_week = df_hist[df_hist['ts'] >= df_hist['ts'].max() - pd.Timedelta(days=7)]
        if last_week.empty:
             print("    [Clima] ADVERTENCIA: No hay datos última semana. Usando dummy.")
             # Llamada recursiva al inicio de la función para generar dummy
             return fetch_future_weather(start_date, end_date)

        last_week_mapping = last_week.set_index('ts')
        for date in future_dates:
            sim_date = date - pd.Timedelta(days=7)
            # Buscar en el índice, necesita manejo exacto de hora
            sim_ts_floor = sim_date.floor('h')
            if sim_ts_floor in last_week_mapping.index:
                 data_sim = last_week_mapping.loc[[sim_ts_floor]].reset_index(drop=True) # Tomar la fila exacta
                 data_sim['ts'] = date # Asignar la fecha futura
                 df_future_list.append(data_sim)

    if not df_future_list:
        print("    [Clima] ADVERTENCIA: Falló simulación. Usando dummy.")
        return fetch_future_weather(start_date, end_date)

    df_simulado = pd.concat(df_future_list)
    all_comunas = df_hist['comuna'].unique()
    all_dates = future_dates
    full_index = pd.MultiIndex.from_product([all_comunas, all_dates], names=['comuna', 'ts'])
    df_final = df_simulado.set_index(['comuna', 'ts']).reindex(full_index)
    df_final = df_final.groupby(level='comuna').ffill().bfill()
    df_final = df_final.fillna(0)

    print(f"    [Clima] Simulación de API completada. {len(df_final)} registros generados.")
    return df_final.reset_index()


def process_future_climate(df_future_weather, df_baselines):
    print("    [Clima] Procesando datos futuros y calculando anomalías...")
    df = normalize_climate_columns(df_future_weather.copy())

    if 'ts' not in df.columns or df['ts'].isnull().all():
         df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    else:
        df["ts"] = df["ts"].dt.tz_convert(TZ)

    df = df.dropna(subset=['ts']).sort_values(['comuna', 'ts'])
    df['dow'] = df['ts'].dt.dayofweek
    df['hour'] = df['ts'].dt.hour

    # --- CORRECCIÓN v14.1 ---
    # Hacer merge usando las columnas correctas (sin _)
    df_merged = pd.merge(
        df,
        df_baselines,
        on=['comuna', 'dow', 'hour'], # <-- Nombres correctos
        how='left'
    )
    # --- FIN CORRECCIÓN ---

    numeric_cols = df_merged.select_dtypes(include=np.number).columns
    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(df_merged[numeric_cols].mean())

    expected_metrics = [c for c in ['temperatura', 'precipitacion', 'lluvia'] if c in df.columns]
    anomaly_cols = []
    for metric in expected_metrics:
        median_col = f'{metric}_median'
        std_col = f'{metric}_std'
        anomaly_col_name = f'anomalia_{metric}'
        if median_col in df_merged.columns and std_col in df_merged.columns:
            df_merged[anomaly_col_name] = (df_merged[metric] - df_merged[median_col]) / (df_merged[std_col] + 1e-6)
        else:
             print(f"    [Clima] ADVERTENCIA: Faltan baselines para {metric}. Anomalía será 0.")
             df_merged[anomaly_col_name] = 0
        anomaly_cols.append(anomaly_col_name)

    df_per_comuna_anomalies = df_merged[['ts', 'comuna'] + anomaly_cols + expected_metrics].copy()
    n_comunas = df_merged['comuna'].nunique()
    if n_comunas == 0: n_comunas = 1

    agg_functions = {}
    for col in anomaly_cols:
        agg_functions[col] = ['max', 'sum', lambda x: (x > 2.5).sum() / n_comunas if n_comunas > 0 else 0]

    if not agg_functions:
        print("    [Clima] ADVERTENCIA: No se generaron funciones de agregación de anomalías.")
        df_agregado = pd.DataFrame(columns=['ts']) # Cambiado a 'ts'
    else:
        df_agregado = df_merged.groupby('ts').agg(agg_functions).reset_index()
        new_cols = ['ts'] # Cambiado a 'ts'
        for col in df_agregado.columns[1:]:
            agg_name = col[1] if col[1] != '<lambda_0>' else 'pct_comunas_afectadas'
            new_cols.append(f"{col[0]}_{agg_name}")
        df_agregado.columns = new_cols

    print("    [Clima] Cálculo de anomalías completado.")
    return df_agregado, df_per_comuna_anomalies


def generate_alerts_json(df_per_comuna, df_risk_proba, proba_threshold=0.5, impact_factor=100):
    print("    [Alertas] Generando 'alertas_climaticas.json'...")
    df_alertas = pd.merge(df_per_comuna, df_risk_proba, on='ts', how='left')
    df_alertas = df_alertas[df_alertas['risk_proba'] > proba_threshold].copy()

    if df_alertas.empty:
        print("    [Alertas] No se detectaron alertas climáticas.")
        return []

    df_alertas['impacto_heuristico'] = (df_alertas['risk_proba'] - proba_threshold) * impact_factor
    df_alertas = df_alertas.sort_values(['comuna', 'ts'])
    json_output = []

    for comuna, group in df_alertas.groupby('comuna'):
        if group.empty: continue
        group['time_diff'] = group['ts'].diff().dt.total_seconds() / 3600
        group['bloque'] = (group['time_diff'] > 1).cumsum()
        for _, bloque_group in group.groupby('bloque'):
            ts_inicio = bloque_group['ts'].min()
            ts_fin = bloque_group['ts'].max() + pd.Timedelta(minutes=59)
            anomalias_dict = {}
            for col in bloque_group.columns:
                if col.startswith('anomalia_'):
                    anomalias_dict[f"{col.replace('anomalia_', '')}_z_max"] = round(bloque_group[col].max(), 2)
            alerta = {
                "comuna": comuna,
                "ts_inicio": ts_inicio.strftime('%Y-%m-%d %H:%M:%S'),
                "ts_fin": ts_fin.strftime('%Y-%m-%d %H:%M:%S'),
                "anomalias": anomalias_dict,
                "impacto_llamadas_adicionales": int(bloque_group['impacto_heuristico'].sum())
            }
            json_output.append(alerta)
    return json_output


# --- FUNCIÓN PRINCIPAL ORQUESTADORA ---

def main(horizonte_dias):
    print("="*60)
    print(f"INICIANDO PIPELINE DE INFERENCIA (v_main 14.1)") # Versión
    print(f"Zona Horaria: {TZ} | Horizonte: {horizonte_dias} días")
    print("="*60)

    # --- 1. Cargar Modelos y Artefactos ---
    print("\n--- Fase 1: Cargando Modelos y Artefactos ---")
    try:
        custom_objects_dict = {'loss': quantile_loss(q=QUANTILE_P)}
        model_planner = tf.keras.models.load_model(PLANNER_MODEL_FILE, custom_objects=custom_objects_dict)
        scaler_planner = joblib.load(PLANNER_SCALER_FILE)
        with open(PLANNER_COLS_FILE, 'r') as f: cols_planner = json.load(f)

        model_risk = tf.keras.models.load_model(RISK_MODEL_FILE)
        scaler_risk = joblib.load(RISK_SCALER_FILE)
        with open(RISK_COLS_FILE, 'r') as f: cols_risk = json.load(f)
        # Cargar baselines de forma segura
        try:
             baselines_clima = joblib.load(RISK_BASELINES_FILE)
        except FileNotFoundError:
             print(f"  [Adv] No se encontró {RISK_BASELINES_FILE}. El pipeline de clima usará medias.")
             baselines_clima = pd.DataFrame() # Crear DF vacío para que no falle


        model_tmo = tf.keras.models.load_model(TMO_MODEL_FILE, custom_objects=custom_objects_dict)
        scaler_tmo = joblib.load(TMO_SCALER_FILE)
        with open(TMO_COLS_FILE, 'r') as f: cols_tmo = json.load(f)

        print("  [OK] Todos los modelos, scalers y columnas cargados.")
    except Exception as e:
        print(f"  [ERROR] Falla crítica al cargar artefactos: {e}")
        print("  Asegúrate de que los archivos existan en 'models/' y provengan del release 'AI-2'.")
        return

    # --- 2. Cargar Datos Históricos ---
    print("\n--- Fase 2: Cargando Datos Históricos ---")
    df_hosting_full = read_data(HOSTING_FILE)
    df_hosting = ensure_ts_and_tz(df_hosting_full)

    try:
        df_feriados_lookup = read_data(FERIADOS_FILE)
        df_feriados_lookup['Fecha_dt'] = pd.to_datetime(df_feriados_lookup['Fecha'], format='%d-%m-%Y', errors='coerce').dt.date
        feriados_list = set(df_feriados_lookup['Fecha_dt'].dropna())
    except Exception as e:
        print(f"  [Adv] No se pudo cargar {FERIADOS_FILE}. Feature 'feriados' será 0. Error: {e}")
        feriados_list = set()

    if 'feriados' not in df_hosting.columns:
        print("  [Info] Creando columna 'feriados'.")
        df_hosting['feriados'] = df_hosting['ts'].dt.date.isin(feriados_list).astype(int)
    else:
        df_hosting['feriados'] = pd.to_numeric(df_hosting['feriados'], errors='coerce').fillna(0).astype(int)

    # Añadir 'dia_despues_feriado' AHORA que 'feriados' está asegurada
    df_hosting['dia_despues_feriado'] = df_hosting['feriados'].shift(24).fillna(0).astype(int)

    if 'recibidos' in df_hosting.columns and TARGET_CALLS not in df_hosting.columns:
         df_hosting = df_hosting.rename(columns={'recibidos': TARGET_CALLS})
    elif TARGET_CALLS not in df_hosting.columns:
        raise ValueError(f"No se encontró la columna {TARGET_CALLS} ni 'recibidos'.")

    df_hosting = df_hosting.groupby("ts").agg({TARGET_CALLS: 'sum', 'feriados': 'max', 'dia_despues_feriado': 'max'}).reset_index() # Agrupar también dia_despues_feriado
    df_hosting = add_time_parts(df_hosting)

    df_tmo_hist = read_data(TMO_FILE)
    df_tmo_hist = ensure_ts_and_tz(df_tmo_hist)
    df_tmo_hist.columns = [c.lower().strip().replace(' ', '_') for c in df_tmo_hist.columns] # Estandarizar TMO cols
    df_tmo_hist = df_tmo_hist.rename(columns={'tmo_general': TARGET_TMO})


    if TARGET_TMO not in df_tmo_hist.columns and all(c in df_tmo_hist.columns for c in ['tmo_comercial', 'q_comercial', 'tmo_tecnico', 'q_tecnico', 'q_general']):
        df_tmo_hist[TARGET_TMO] = (df_tmo_hist['tmo_comercial'] * df_tmo_hist['q_comercial'] + df_tmo_hist['tmo_tecnico'] * df_tmo_hist['q_tecnico']) / (df_tmo_hist['q_general'] + 1e-6)

    if 'q_llamadas_comercial' in df_tmo_hist.columns and 'q_llamadas_general' in df_tmo_hist.columns:
        df_tmo_hist['proporcion_comercial'] = df_tmo_hist['q_llamadas_comercial'] / (df_tmo_hist['q_llamadas_general'] + 1e-6)
        df_tmo_hist['proporcion_tecnica'] = df_tmo_hist['q_llamadas_tecnico'] / (df_tmo_hist['q_llamadas_general'] + 1e-6)
    else:
        print("  [Adv] Columnas q_llamadas no encontradas en TMO. Usando proporciones=0.")
        df_tmo_hist['proporcion_comercial'] = 0
        df_tmo_hist['proporcion_tecnica'] = 0

    last_hist_ts = df_hosting['ts'].max()
    print(f"  [OK] Datos históricos cargados. Último timestamp usado como 'seed': {last_hist_ts}")

    # --- 3. Generar Esqueleto Futuro ---
    print("\n--- Fase 3: Generando Esqueleto de Fechas Futuras ---")
    start_future = last_hist_ts + pd.Timedelta(hours=1)
    # Ajustar end_future para asegurar que cubra el horizonte completo
    end_future = start_future + pd.Timedelta(days=horizonte_dias, hours=23)


    df_future = pd.DataFrame(
        pd.date_range(start=start_future, end=end_future, freq='h', tz=TZ),
        columns=['ts']
    )
    # Recortar al horizonte exacto en horas si es necesario (ej, si horizonte_dias=0)
    df_future = df_future.iloc[:horizonte_dias * 24]


    df_future = add_time_parts(df_future)
    df_future['feriados'] = df_future['ts'].dt.date.isin(feriados_list).astype(int)
    # Calcular dia_despues_feriado para el futuro
    # Necesitamos mirar el día anterior (24h atrás) que podría estar en el histórico
    temp_ts_series = pd.concat([df_hosting['ts'].iloc[-24:], df_future['ts']])
    temp_feriados_series = pd.concat([df_hosting['feriados'].iloc[-24:], df_future['feriados']])
    all_feriados_df = pd.DataFrame({'ts': temp_ts_series, 'feriados': temp_feriados_series}).set_index('ts')
    future_feriados_shifted = all_feriados_df['feriados'].shift(24).loc[df_future['ts']]
    df_future['dia_despues_feriado'] = future_feriados_shifted.fillna(0).astype(int)

    print(f"  [OK] Esqueleto futuro creado desde {df_future['ts'].min()} hasta {df_future['ts'].max()}")


    # --- 4. Pipeline Clima (Analista de Riesgos) ---
    print("\n--- Fase 4: Pipeline de Clima (Analista de Riesgos) ---")
    df_weather_future_raw = fetch_future_weather(start_future, end_future)

    # Pasar baselines vacíos si no se cargaron
    df_agg_anomalies, df_per_comuna_anomalies = process_future_climate(df_weather_future_raw, baselines_clima if not baselines_clima.empty else pd.DataFrame())

    df_future = pd.merge(df_future, df_agg_anomalies, on='ts', how='left')
    numeric_cols_future = df_future.select_dtypes(include=np.number).columns
    df_future[numeric_cols_future] = df_future[numeric_cols_future].fillna(df_future[numeric_cols_future].mean())
    df_future = df_future.fillna(0)

    # Predecir Riesgo (solo si las columnas existen)
    if all(c in df_future.columns for c in cols_risk):
        X_risk = df_future.reindex(columns=cols_risk, fill_value=0)
        X_risk_s = scaler_risk.transform(X_risk)
        df_future['risk_proba'] = model_risk.predict(X_risk_s)
        print("  [OK] Predicciones de riesgo generadas.")
    else:
        print("  [Adv] Faltan columnas para predecir riesgo. 'risk_proba' será 0.")
        df_future['risk_proba'] = 0.0

    df_risk_proba_output = df_future[['ts', 'risk_proba']].copy()
    alertas_json_data = generate_alerts_json(df_per_comuna_anomalies, df_risk_proba_output)

    # --- 5. Pipeline Llamadas (Planificador) ---
    print("\n--- Fase 5: Pipeline de Llamadas (Planificador) ---")
    df_full = pd.concat([df_hosting, df_future], ignore_index=True).sort_values('ts')
    for lag in [24, 48, 72, 168]:
        df_full[f'lag_{lag}'] = df_full[TARGET_CALLS].shift(lag)
    for window in [24, 72, 168]:
        df_full[f'ma_{window}'] = df_full[TARGET_CALLS].shift(1).rolling(window, min_periods=1).mean()

    df_future_features = df_full[df_full['ts'] >= start_future].copy()
    X_planner = pd.get_dummies(df_future_features, columns=['month', 'semana_del_mes'])
    # Asegurar que todas las columnas existan, rellenando con 0
    X_planner = X_planner.reindex(columns=cols_planner, fill_value=0)

    numeric_cols_planner = X_planner.select_dtypes(include=np.number).columns
    # Rellenar NaNs con la media de la columna si existe, sino con 0
    means = X_planner[numeric_cols_planner].mean()
    X_planner[numeric_cols_planner] = X_planner[numeric_cols_planner].fillna(means).fillna(0)

    X_planner_s = scaler_planner.transform(X_planner)
    df_future['llamadas_hora'] = model_planner.predict(X_planner_s).clip(0).astype(int)
    print("  [OK] Predicciones de llamadas (Planificador) generadas.")

    # --- 6. Pipeline de TMO (Analista de Operaciones) ---
    print("\n--- Fase 6: Pipeline de TMO (Analista de Operaciones) ---")
    if df_tmo_hist.empty:
        print("  [Adv] TMO_HISTORICO.csv vacío. Usando TMO=0.")
        last_tmo_data = pd.Series(dtype='float64')
    else:
        last_tmo_data = df_tmo_hist.sort_values('ts').iloc[-1]

    seed_cols = ['proporcion_comercial', 'proporcion_tecnica', 'tmo_comercial', 'tmo_tecnico']
    df_tmo_features = df_future.copy()
    df_tmo_features[TARGET_CALLS] = df_tmo_features['llamadas_hora']
    for col in seed_cols:
        if col in last_tmo_data and pd.notna(last_tmo_data[col]):
            df_tmo_features[col] = last_tmo_data[col]
        else:
            df_tmo_features[col] = 0

    X_tmo = pd.get_dummies(df_tmo_features, columns=['month', 'semana_del_mes'])
    X_tmo = X_tmo.reindex(columns=cols_tmo, fill_value=0)
    numeric_cols_tmo = X_tmo.select_dtypes(include=np.number).columns
    means_tmo = X_tmo[numeric_cols_tmo].mean()
    X_tmo[numeric_cols_tmo] = X_tmo[numeric_cols_tmo].fillna(means_tmo).fillna(0)


    X_tmo_s = scaler_tmo.transform(X_tmo)
    df_future['tmo_hora'] = model_tmo.predict(X_tmo_s).clip(0)
    print("  [OK] Predicciones de TMO generadas.")

    # --- 7. Generar Salidas Finales (JSONs) ---
    print("\n--- Fase 7: Generando Archivos JSON de Salida ---")
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    df_future['agentes_requeridos'] = calculate_erlang_agents(df_future['llamadas_hora'], df_future['tmo_hora'])
    df_horaria = df_future[['ts', 'llamadas_hora', 'tmo_hora', 'agentes_requeridos']].copy()
    df_horaria['ts'] = df_horaria['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    output_path_horaria = os.path.join(PUBLIC_DIR, "prediccion_horaria.json")
    df_horaria.to_json(output_path_horaria, orient='records', indent=2, force_ascii=False)
    print(f"  [OK] Archivo 'prediccion_horaria.json' guardado.")

    df_horaria_para_diaria = df_future.copy()
    df_horaria_para_diaria['fecha'] = df_horaria_para_diaria['ts'].dt.date
    df_horaria_para_diaria['tmo_ponderado_num'] = df_horaria_para_diaria['tmo_hora'] * df_horaria_para_diaria['llamadas_hora']
    df_diaria_agg = df_horaria_para_diaria.groupby('fecha').agg(
        llamadas_totales_dia=('llamadas_hora', 'sum'),
        tmo_ponderado_num=('tmo_ponderado_num', 'sum')
    )
    df_diaria_agg['tmo_promedio_diario'] = df_diaria_agg['tmo_ponderado_num'] / (df_diaria_agg['llamadas_totales_dia'] + 1e-6)
    if (df_diaria_agg['llamadas_totales_dia'] == 0).any():
        tmo_simple_avg = df_horaria_para_diaria.groupby('fecha')['tmo_hora'].mean()
        df_diaria_agg['tmo_promedio_diario'] = df_diaria_agg['tmo_promedio_diario'].where(
            df_diaria_agg['llamadas_totales_dia'] > 0,
            tmo_simple_avg
        )
    df_diaria_agg = df_diaria_agg.reset_index()[['fecha', 'llamadas_totales_dia', 'tmo_promedio_diario']]
    df_diaria_agg['fecha'] = df_diaria_agg['fecha'].astype(str)
    df_diaria_agg['llamadas_totales_dia'] = df_diaria_agg['llamadas_totales_dia'].astype(int)
    output_path_diaria = os.path.join(PUBLIC_DIR, "Predicion_daria.json")
    df_diaria_agg.to_json(output_path_diaria, orient='records', indent=2, force_ascii=False)
    print(f"  [OK] Archivo 'Predicion_daria.json' guardado.")

    output_path_alertas = os.path.join(PUBLIC_DIR, "alertas_climaticas.json")
    with open(output_path_alertas, 'w', encoding='utf-8') as f:
        json.dump(alertas_json_data, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Archivo 'alertas_climaticas.json' guardado.")

    print("\n" + "="*60)
    print("PIPELINE DE INFERENCIA COMPLETADO EXITOSAMENTE.")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Inferencia IA para Predicción de Tráfico.")
    parser.add_argument(
        "--horizonte",
        type=int,
        default=120,
        help="Horizonte de predicción en días (default: 120)"
    )
    args = parser.parse_args()

    main(horizonte_dias=args.horizonte)
