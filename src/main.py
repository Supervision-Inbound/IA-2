Run python -m src.main --horizonte 120
2025-10-19 23:21:19.777441: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-10-19 23:21:19.821818: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-10-19 23:21:22.194850: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-10-19 23:21:22.764661: E external/local_xla/xla/stream_executor/cuda/cuda_platform.cc:51] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
/opt/hostedtoolcache/Python/3.10.19/x64/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.2.2 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
/opt/hostedtoolcache/Python/3.10.19/x64/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.2.2 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
2025-10-19 23:21:24.360978: I external/local_xla/xla/service/service.cc:163] XLA service 0x7f46d80079a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2025-10-19 23:21:24.360994: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (0): Host, Default Version
2025-10-19 23:21:24.366044: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1760926884.413720    2157 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
============================================================
INICIANDO PIPELINE DE INFERENCIA (réplica planner iterativo + feriados)
Zona Horaria: America/Santiago | Horizonte: 120 días
============================================================

--- Fase 1: Cargando Modelos y Artefactos (v7) ---
  [OK] Todos los modelos v7 cargados.

--- Fase 2: Cargando Datos Históricos ---
  [OK] Datos históricos cargados. Último timestamp: 2025-07-27 22:00:00-04:00

--- Fase 3: Generando Esqueleto de Fechas Futuras ---
  [OK] Esqueleto futuro creado: 2025-07-27 23:00:00-04:00 a 2025-11-24 23:00:00-03:00

--- Fase 4: Pipeline de Clima (Analista de Riesgos) ---
    [Clima] SIMULANDO API de clima futuro...
    [Clima] ADVERTENCIA: No cols clima en /home/runner/work/IA-2/IA-2/data/historical_data.csv. Dummy.
    [Clima] Procesando datos futuros y calculando anomalías...
    [Clima] Cálculo anomalías completado.

 1/90 ━━━━━━━━━━━━━━━━━━━━ 9s 111ms/step
71/90 ━━━━━━━━━━━━━━━━━━━━ 0s 716us/step
90/90 ━━━━━━━━━━━━━━━━━━━━ 0s 822us/step
Traceback (most recent call last):
  File "/home/runner/work/IA-2/IA-2/src/main.py", line 48, in _ensure_local_tz
    if s.dt.tz is None:
AttributeError: 'DatetimeIndex' object has no attribute 'dt'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.10.19/x64/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/hostedtoolcache/Python/3.10.19/x64/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/runner/work/IA-2/IA-2/src/main.py", line 691, in <module>
    main(horizonte_dias=args.horizonte)
  File "/home/runner/work/IA-2/IA-2/src/main.py", line 581, in main
    tmp = add_time_parts(tmp)
  File "/home/runner/work/IA-2/IA-2/src/main.py", line 259, in add_time_parts
    s = _ensure_local_tz(pd.Index(df_copy.index))
  File "/home/runner/work/IA-2/IA-2/src/main.py", line 54, in _ensure_local_tz
    s = pd.to_datetime(s, utc=True, errors='coerce').dt.tz_convert(TZ)
AttributeError: 'DatetimeIndex' object has no attribute 'dt'
  [OK] Predicciones riesgo generadas.
    [Alertas] Generando 'alertas_climaticas.json'...
    [Alertas] No se detectaron alertas.

--- Fase 5: Planner iterativo de Llamadas (réplica) ---
Error: Process completed with exit code 1.
