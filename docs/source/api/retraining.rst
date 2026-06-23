retraining
==========

.. automodule:: src.retraining
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Sistema de re-entrenamiento automático de modelos con detección de data drift, versionado de modelos y comparación de rendimiento.

Clases Principales
------------------

ModelRetrainer
^^^^^^^^^^^^^^

.. autoclass:: src.retraining.ModelRetrainer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Sistema completo de re-entrenamiento con gestión de versiones y detección de drift.
   
   **Atributos:**
   
   * ``models_dir``: Directorio de modelos activos (models/supervised/)
   * ``versions_dir``: Directorio de versiones (models/versions/)
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.retraining import ModelRetrainer
      import pandas as pd
      
      # Crear retrainer
      retrainer = ModelRetrainer()
      
      # Ver modelos disponibles
      models = retrainer.get_available_models()
      print(f"Modelos disponibles: {models}")
      
      # Cargar información del modelo
      info = retrainer.load_model_info('xgboost')
      print(f"Accuracy actual: {info['metrics']['accuracy']:.3f}")
      
      # Cargar nuevos datos
      new_data = pd.read_csv("nuevos_datos.csv")
      
      # Detectar drift
      original_data = pd.read_csv("datos_originales.csv")
      drift_results = retrainer.detect_data_drift(
          original_data,
          new_data,
          info['feature_names']
      )
      
      print(f"Características con drift: {len(drift_results['features_with_drift'])}")
      
      # Crear backup
      version_name = retrainer.create_version_backup('xgboost')
      print(f"Backup creado: {version_name}")
      
      # Re-entrenar
      results = retrainer.retrain_model('xgboost', new_data)
      print(f"Nueva accuracy: {results['metrics']['accuracy']:.3f}")

Métodos de Gestión
------------------

get_available_models
^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.retraining.ModelRetrainer.get_available_models
   :noindex:

Obtiene lista de modelos disponibles para re-entrenamiento.

**Returns:**
   Lista de claves de modelos

**Ejemplo:**

.. code-block:: python

   models = retrainer.get_available_models()
   
   for model_key in models:
       print(f"- {model_key}")

load_model_info
^^^^^^^^^^^^^^^

.. automethod:: src.retraining.ModelRetrainer.load_model_info
   :noindex:

Carga información completa del modelo.

**Parameters:**
   * ``model_key`` (str): Clave del modelo

**Returns:**
   Diccionario con información del modelo

**Información incluida:**
   * ``model_data``: Datos del modelo (pickle)
   * ``timestamp``: Fecha de entrenamiento
   * ``feature_names``: Características usadas
   * ``metrics``: Métricas de evaluación

**Ejemplo:**

.. code-block:: python

   info = retrainer.load_model_info('random_forest')
   
   print(f"Timestamp: {info['timestamp']}")
   print(f"Características: {len(info['feature_names'])}")
   print(f"Accuracy: {info['metrics']['accuracy']:.4f}")
   print(f"F1-Score: {info['metrics']['f1_weighted']:.4f}")

Métodos de Detección de Drift
------------------------------

detect_data_drift
^^^^^^^^^^^^^^^^^

.. automethod:: src.retraining.ModelRetrainer.detect_data_drift
   :noindex:

Detecta data drift entre datasets usando tests estadísticos.

**Parameters:**
   * ``original_data`` (DataFrame): Dataset original
   * ``new_data`` (DataFrame): Dataset nuevo
   * ``feature_names`` (List[str]): Características a comparar

**Returns:**
   Diccionario con resultados del análisis

**Tests estadísticos aplicados:**

1. **Kolmogorov-Smirnov (KS):** Compara distribuciones completas
2. **Mann-Whitney U:** Compara medianas
3. **Diferencia de medias:** Cambio en tendencia central
4. **Diferencia de desviaciones:** Cambio en dispersión

**Criterio de drift:**
   * p-value < 0.05 en KS o Mann-Whitney
   * Indica cambio significativo en la distribución

**Ejemplo:**

.. code-block:: python

   # Cargar datos
   df_original = pd.read_csv("datos_2023.csv")
   df_new = pd.read_csv("datos_2024.csv")
   
   # Detectar drift
   drift_results = retrainer.detect_data_drift(
       df_original,
       df_new,
       feature_names=['edad', 'salario_mensual', 'puntaje_datacredito', 'dti']
   )
   
   # Analizar resultados
   print(f"Características con drift: {len(drift_results['features_with_drift'])}")
   
   for feature in drift_results['features_with_drift']:
       scores = drift_results['drift_scores'][feature]
       print(f"\n{feature}:")
       print(f"  KS p-value: {scores['ks_pvalue']:.6f}")
       print(f"  Diferencia media: {scores['mean_diff']:.4f}")
       print(f"  Diferencia std: {scores['std_diff']:.4f}")
   
   # Decisión
   drift_percentage = len(drift_results['features_with_drift']) / len(feature_names) * 100
   
   if drift_percentage > 30:
       print("\n⚠️ RECOMENDACIÓN: Re-entrenar modelo (>30% características con drift)")
   elif drift_percentage > 10:
       print("\n💡 RECOMENDACIÓN: Considerar re-entrenamiento (10-30% drift)")
   else:
       print("\n✅ RECOMENDACIÓN: Modelo actual es adecuado (<10% drift)")

Métodos de Versionado
----------------------

create_version_backup
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.retraining.ModelRetrainer.create_version_backup
   :noindex:

Crea backup versionado del modelo actual.

**Parameters:**
   * ``model_key`` (str): Clave del modelo

**Returns:**
   Nombre de la versión creada

**Formato de versión:**
   ``{model_key}_v{YYYYMMDD_HHMMSS}``

**Archivos guardados:**
   * ``{version_name}_model.pkl``: Modelo completo
   * ``{version_name}_metrics.json``: Métricas de evaluación

**Ejemplo:**

.. code-block:: python

   # Crear backup antes de re-entrenar
   version_name = retrainer.create_version_backup('xgboost')
   
   print(f"✅ Backup creado: {version_name}")
   print(f"   Ubicación: models/versions/{version_name}_model.pkl")
   
   # Ahora es seguro re-entrenar
   results = retrainer.retrain_model('xgboost', new_data)

Métodos de Re-entrenamiento
----------------------------

retrain_model
^^^^^^^^^^^^^

.. automethod:: src.retraining.ModelRetrainer.retrain_model
   :noindex:

Re-entrena un modelo con nuevos datos.

**Parameters:**
   * ``model_key`` (str): Clave del modelo a re-entrenar
   * ``new_data`` (DataFrame): Nuevos datos para entrenamiento
   * ``target_col`` (str): Variable objetivo (default: 'nivel_riesgo')

**Returns:**
   Diccionario con resultados del re-entrenamiento

**Proceso:**

1. Crear instancia de SupervisedModelTrainer
2. Preparar nuevos datos (70/20/10 split)
3. Re-entrenar con GridSearchCV
4. Calcular métricas completas
5. Guardar modelo actualizado
6. Retornar resultados

**Ejemplo:**

.. code-block:: python

   # Cargar nuevos datos
   new_data = pd.read_csv("datos_nuevos_2024.csv")
   
   # Re-entrenar
   results = retrainer.retrain_model(
       model_key='xgboost',
       new_data=new_data,
       target_col='nivel_riesgo'
   )
   
   # Ver resultados
   print(f"Modelo re-entrenado exitosamente")
   print(f"Nuevas métricas:")
   print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
   print(f"  F1-Score: {results['metrics']['f1_weighted']:.4f}")
   print(f"  ROC-AUC: {results['metrics']['roc_auc']:.4f}")
   
   # Mejores parámetros encontrados
   print(f"\nMejores hiperparámetros:")
   for param, value in results['best_params'].items():
       print(f"  {param}: {value}")

Funciones de Renderizado
-------------------------

render_retraining_module
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.retraining.render_retraining_module

Renderiza el módulo completo de re-entrenamiento en Streamlit.

**Funcionalidades:**

1. **Selección de modelo:**
   
   * Lista de modelos disponibles
   * Métricas actuales
   * Fecha de última actualización

2. **Carga de datos:**
   
   * Usar datos existentes
   * Cargar archivo nuevo (CSV/Excel)
   * Vista previa de datos

3. **Detección de drift:**
   
   * Tests estadísticos (KS, Mann-Whitney)
   * Visualización de características con drift
   * Recomendaciones automáticas

4. **Re-entrenamiento:**
   
   * Opción de crear backup
   * Optimización de hiperparámetros
   * Comparación de rendimiento
   * Gráficos comparativos

5. **Gestión de versiones:**
   
   * Lista de versiones guardadas
   * Restauración de versiones anteriores

Tests Estadísticos
------------------

Test de Kolmogorov-Smirnov
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compara dos distribuciones empíricas.

**Hipótesis nula:** Las dos muestras provienen de la misma distribución

.. math::

   D = \max_x |F_1(x) - F_2(x)|

* **p-value < 0.05:** Rechazar H₀ (hay drift)
* **p-value ≥ 0.05:** No rechazar H₀ (sin drift)

Test de Mann-Whitney U
^^^^^^^^^^^^^^^^^^^^^^^

Compara medianas de dos grupos independientes.

**Hipótesis nula:** Las medianas son iguales

* **p-value < 0.05:** Medianas diferentes (hay drift)
* **p-value ≥ 0.05:** Medianas similares (sin drift)

Ejemplo Completo de Re-entrenamiento
-------------------------------------

.. code-block:: python

   from src.retraining import ModelRetrainer
   import pandas as pd
   
   # Inicializar
   retrainer = ModelRetrainer()
   
   # Paso 1: Cargar modelo actual
   model_key = 'xgboost'
   model_info = retrainer.load_model_info(model_key)
   
   print(f"Modelo actual: {model_key}")
   print(f"Accuracy: {model_info['metrics']['accuracy']:.4f}")
   print(f"Fecha: {model_info['timestamp']}")
   
   # Paso 2: Cargar datos
   df_original = pd.read_csv("data/processed/datos_credito_hipotecario_realista.csv")
   df_new = pd.read_csv("data/new/datos_nuevos_2024.csv")
   
   print(f"\nDatos originales: {len(df_original):,} registros")
   print(f"Datos nuevos: {len(df_new):,} registros")
   
   # Paso 3: Detectar drift
   drift_results = retrainer.detect_data_drift(
       df_original,
       df_new,
       model_info['feature_names']
   )
   
   n_drift = len(drift_results['features_with_drift'])
   total_features = len(model_info['feature_names'])
   drift_pct = (n_drift / total_features) * 100
   
   print(f"\nAnálisis de Drift:")
   print(f"  Características con drift: {n_drift}/{total_features} ({drift_pct:.1f}%)")
   
   if drift_results['features_with_drift']:
       print(f"\n  Características afectadas:")
       for feature in drift_results['features_with_drift'][:5]:
           scores = drift_results['drift_scores'][feature]
           print(f"    - {feature}: KS p-value={scores['ks_pvalue']:.4f}")
   
   # Paso 4: Decidir si re-entrenar
   should_retrain = drift_pct > 20  # Umbral: 20%
   
   if should_retrain:
       print(f"\n⚠️ Se recomienda re-entrenar (drift > 20%)")
       
       # Paso 5: Crear backup
       version_name = retrainer.create_version_backup(model_key)
       print(f"✅ Backup creado: {version_name}")
       
       # Paso 6: Re-entrenar
       print(f"\n🔄 Re-entrenando modelo...")
       results = retrainer.retrain_model(model_key, df_new)
       
       # Paso 7: Comparar rendimiento
       old_accuracy = model_info['metrics']['accuracy']
       new_accuracy = results['metrics']['accuracy']
       improvement = new_accuracy - old_accuracy
       
       print(f"\n📊 Comparación de Rendimiento:")
       print(f"  Accuracy anterior: {old_accuracy:.4f}")
       print(f"  Accuracy nueva: {new_accuracy:.4f}")
       print(f"  Mejora: {improvement:+.4f}")
       
       if improvement > 0.01:
           print(f"\n✅ Modelo mejorado significativamente")
       elif improvement < -0.01:
           print(f"\n⚠️ Modelo empeoró - considerar restaurar backup")
       else:
           print(f"\n➖ Rendimiento similar")
   else:
       print(f"\n✅ No se requiere re-entrenamiento (drift < 20%)")

Workflow de Re-entrenamiento
-----------------------------

Flujo Recomendado
^^^^^^^^^^^^^^^^^

.. code-block:: text

   1. Cargar modelo actual
      ↓
   2. Cargar nuevos datos
      ↓
   3. Detectar data drift
      ↓
   4. ¿Drift > 20%? ──No──→ Mantener modelo actual
      │                      
      Sí
      ↓
   5. Crear backup versionado
      ↓
   6. Re-entrenar modelo
      ↓
   7. Comparar rendimiento
      ↓
   8. ¿Mejora > 1%? ──No──→ Restaurar backup
      │
      Sí
      ↓
   9. Usar nuevo modelo

Criterios de Re-entrenamiento
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cuándo re-entrenar:**

* Drift > 20% de características
* Degradación de métricas en producción
* Nuevos datos disponibles (>1000 registros)
* Cambios en el negocio o regulaciones
* Periodicidad: Cada 3-6 meses

**Cuándo NO re-entrenar:**

* Drift < 10% de características
* Pocos datos nuevos (<500 registros)
* Modelo funcionando bien en producción
* Cambios recientes (< 1 mes)

Gestión de Versiones
---------------------

Estructura de Versiones
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   models/
   ├── supervised/           # Modelos activos
   │   ├── xgboost_model.pkl
   │   ├── xgboost_metrics.json
   │   └── ...
   └── versions/            # Versiones históricas
       ├── xgboost_v20240115_143022_model.pkl
       ├── xgboost_v20240115_143022_metrics.json
       ├── xgboost_v20240220_091545_model.pkl
       └── ...

Restaurar Versión Anterior
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import shutil
   from pathlib import Path
   
   # Listar versiones disponibles
   versions_dir = Path("models/versions")
   version_files = sorted(versions_dir.glob("xgboost_v*_model.pkl"), reverse=True)
   
   print("Versiones disponibles:")
   for i, version_file in enumerate(version_files):
       version_name = version_file.stem.replace('_model', '')
       print(f"{i+1}. {version_name}")
   
   # Restaurar versión específica
   version_to_restore = version_files[0]  # Más reciente
   
   shutil.copy2(
       version_to_restore,
       "models/supervised/xgboost_model.pkl"
   )
   
   print(f"✅ Versión restaurada: {version_to_restore.stem}")

Monitoreo de Drift
-------------------

Métricas de Drift
^^^^^^^^^^^^^^^^^

Para cada característica se calcula:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Métrica
     - Descripción
   * - KS Statistic
     - Máxima diferencia entre CDFs
   * - KS p-value
     - Significancia estadística
   * - MW p-value
     - Test de Mann-Whitney
   * - Mean Diff
     - \|media_nueva - media_original\|
   * - Std Diff
     - \|std_nueva - std_original\|

Visualización de Drift
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import plotly.graph_objects as go
   
   # Crear gráfico de drift
   features = list(drift_results['drift_scores'].keys())
   ks_pvalues = [drift_results['drift_scores'][f]['ks_pvalue'] for f in features]
   
   fig = go.Figure()
   
   fig.add_trace(go.Bar(
       x=features,
       y=ks_pvalues,
       marker_color=['red' if p < 0.05 else 'green' for p in ks_pvalues]
   ))
   
   fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                 annotation_text="Umbral de significancia")
   
   fig.update_layout(
       title="Data Drift por Característica",
       xaxis_title="Características",
       yaxis_title="KS p-value",
       template="plotly_white"
   )
   
   fig.show()

Ejemplo de Pipeline Completo
-----------------------------

.. code-block:: python

   from src.retraining import ModelRetrainer
   from src.supervised_models import SupervisedModelTrainer
   import pandas as pd
   from datetime import datetime
   
   def automated_retraining_pipeline(model_key: str, new_data_path: str):
       """Pipeline automatizado de re-entrenamiento"""
       
       print(f"{'='*60}")
       print(f"PIPELINE DE RE-ENTRENAMIENTO AUTOMÁTICO")
       print(f"{'='*60}")
       print(f"Modelo: {model_key}")
       print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
       
       # Inicializar
       retrainer = ModelRetrainer()
       
       # 1. Cargar modelo actual
       print(f"\n[1/7] Cargando modelo actual...")
       model_info = retrainer.load_model_info(model_key)
       current_accuracy = model_info['metrics']['accuracy']
       print(f"      Accuracy actual: {current_accuracy:.4f}")
       
       # 2. Cargar datos
       print(f"\n[2/7] Cargando datos...")
       df_original = pd.read_csv("data/processed/datos_credito_hipotecario_realista.csv")
       df_new = pd.read_csv(new_data_path)
       print(f"      Original: {len(df_original):,} registros")
       print(f"      Nuevo: {len(df_new):,} registros")
       
       # 3. Detectar drift
       print(f"\n[3/7] Detectando data drift...")
       drift_results = retrainer.detect_data_drift(
           df_original,
           df_new,
           model_info['feature_names']
       )
       
       n_drift = len(drift_results['features_with_drift'])
       drift_pct = (n_drift / len(model_info['feature_names'])) * 100
       print(f"      Drift detectado: {n_drift} características ({drift_pct:.1f}%)")
       
       # 4. Decidir si re-entrenar
       print(f"\n[4/7] Evaluando necesidad de re-entrenamiento...")
       
       if drift_pct < 10:
           print(f"      ✅ Drift bajo ({drift_pct:.1f}%) - No se requiere re-entrenamiento")
           return {'status': 'skipped', 'reason': 'low_drift', 'drift_pct': drift_pct}
       
       print(f"      ⚠️ Drift significativo ({drift_pct:.1f}%) - Procediendo con re-entrenamiento")
       
       # 5. Crear backup
       print(f"\n[5/7] Creando backup...")
       version_name = retrainer.create_version_backup(model_key)
       print(f"      ✅ Backup: {version_name}")
       
       # 6. Re-entrenar
       print(f"\n[6/7] Re-entrenando modelo...")
       results = retrainer.retrain_model(model_key, df_new)
       new_accuracy = results['metrics']['accuracy']
       improvement = new_accuracy - current_accuracy
       print(f"      Nueva accuracy: {new_accuracy:.4f} ({improvement:+.4f})")
       
       # 7. Validar mejora
       print(f"\n[7/7] Validando mejora...")
       
       if improvement > 0.01:
           print(f"      ✅ Mejora significativa - Modelo actualizado")
           status = 'success'
       elif improvement < -0.01:
           print(f"      ❌ Degradación detectada - Restaurando backup")
           # Restaurar backup aquí
           status = 'degraded'
       else:
           print(f"      ➖ Rendimiento similar - Modelo actualizado")
           status = 'similar'
       
       print(f"\n{'='*60}")
       print(f"PIPELINE COMPLETADO")
       print(f"{'='*60}")
       
       return {
           'status': status,
           'drift_pct': drift_pct,
           'old_accuracy': current_accuracy,
           'new_accuracy': new_accuracy,
           'improvement': improvement,
           'version_backup': version_name
       }
   
   # Ejecutar pipeline
   result = automated_retraining_pipeline(
       model_key='xgboost',
       new_data_path='data/new/datos_2024_q1.csv'
   )
   
   print(f"\nResultado final: {result}")

Mejores Prácticas
-----------------

Frecuencia de Re-entrenamiento
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Mensual:** Si hay flujo constante de datos nuevos
* **Trimestral:** Para la mayoría de casos
* **Semestral:** Si los datos son estables
* **Ad-hoc:** Cuando se detecta degradación

Validación Post-Reentrenamiento
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Comparar métricas:** Accuracy, F1, ROC-AUC
2. **Validar en holdout:** Datos no vistos
3. **A/B testing:** Comparar en producción
4. **Monitorear predicciones:** Primeras semanas

Gestión de Versiones
^^^^^^^^^^^^^^^^^^^^^

* Mantener últimas 10 versiones
* Documentar cambios en cada versión
* Etiquetar versiones en producción
* Backup antes de cada re-entrenamiento

Ver también
-----------

* :doc:`supervised_models` - Entrenamiento de modelos
* :doc:`prediction` - Sistema de predicción
* :doc:`data_processor` - Procesamiento de datos