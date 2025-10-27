prediction
==========

.. automodule:: src.prediction
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Sistema de predicción de riesgo crediticio para nuevos solicitantes con formulario interactivo, explicaciones detalladas y recomendaciones automáticas.

Clases Principales
------------------

CreditRiskPredictor
^^^^^^^^^^^^^^^^^^^

.. autoclass:: src.prediction.CreditRiskPredictor
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Predictor de riesgo crediticio que carga modelos entrenados y realiza predicciones para nuevos solicitantes.
   
   **Atributos:**
   
   * ``available_models``: Diccionario de modelos disponibles
   * ``selected_model``: Modelo actualmente seleccionado
   * ``model_data``: Datos del modelo cargado
   * ``feature_engineer``: Ingeniero de características
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.prediction import CreditRiskPredictor
      
      # Crear predictor
      predictor = CreditRiskPredictor()
      
      # Ver modelos disponibles
      print("Modelos disponibles:")
      for key, info in predictor.available_models.items():
           print(f"  - {info['name']}: Accuracy={info['metrics'].get('accuracy', 0):.3f}")
      
      # Cargar modelo
      predictor.load_model('xgboost')
      
      # Datos del solicitante
      applicant_data = {
          'edad': 35,
          'salario_mensual': 4500000,
          'puntaje_datacredito': 720,
          'valor_inmueble': 180000000,
          'porcentaje_cuota_inicial': 20,
          'plazo_credito': 20,
          'tasa_interes_anual': 11.5,
          # ... más campos
      }
      
      # Calcular características derivadas
      enhanced_data = predictor.calculate_derived_features(applicant_data)
      
      # Predecir riesgo
      results = predictor.predict_risk(enhanced_data)
      
      print(f"Riesgo predicho: {results['predicted_class']}")
      print(f"Probabilidades: {results['probabilities']}")
      print(f"Recomendación: {results['recommendation']['decision']}")

Métodos de Carga
----------------

_load_available_models
^^^^^^^^^^^^^^^^^^^^^^

Carga lista de modelos disponibles desde el directorio de modelos.

**Modelos detectados:**
   * Archivos .pkl en ``models/supervised/``
   * Métricas asociadas en archivos .json

load_model
^^^^^^^^^^

.. automethod:: src.prediction.CreditRiskPredictor.load_model
   :noindex:

Carga un modelo entrenado específico.

**Parameters:**
   * ``model_key`` (str): Clave del modelo a cargar

**Returns:**
   True si se cargó exitosamente

**Ejemplo:**

.. code-block:: python

   # Cargar modelo XGBoost
   success = predictor.load_model('xgboost')
   
   if success:
       print("✅ Modelo cargado")
       print(f"Características requeridas: {len(predictor.model_data['feature_names'])}")
   else:
       print("❌ Error cargando modelo")

Métodos de Formulario
----------------------

create_prediction_form
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.prediction.CreditRiskPredictor.create_prediction_form
   :noindex:

Crea formulario interactivo para capturar datos del solicitante.

**Returns:**
   Diccionario con datos del formulario

**Secciones del formulario:**

1. **👤 Personal:** Edad, estado civil, educación, ciudad, estrato, personas a cargo
2. **💼 Laboral:** Tipo empleo, antigüedad, salario, egresos
3. **💰 Financiero:** Puntaje DataCrédito, patrimonio, propiedades, saldo banco, demandas
4. **🏠 Inmueble:** Valor, cuota inicial, años, plazo, tasa de interés

**Ejemplo de uso en Streamlit:**

.. code-block:: python

   # En aplicación Streamlit
   form_data = predictor.create_prediction_form()
   
   # Los datos se capturan interactivamente
   # form_data contendrá todos los campos del formulario

calculate_derived_features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.prediction.CreditRiskPredictor.calculate_derived_features
   :noindex:

Calcula características derivadas a partir de los datos del formulario.

**Parameters:**
   * ``form_data`` (Dict): Datos del formulario

**Returns:**
   Datos con características derivadas

**Características calculadas:**

* ``valor_cuota_inicial``: Valor × Porcentaje/100
* ``monto_credito``: Valor - Cuota inicial
* ``cuota_mensual``: Calculada con fórmula de amortización
* ``ltv``: (Monto/Valor) × 100
* ``dti``: (Cuota/Salario) × 100
* ``capacidad_ahorro``: Salario - Egresos
* ``capacidad_residual``: Ahorro - Cuota

**Fórmula de cuota mensual:**

.. math::

   C = M \times \frac{i(1+i)^n}{(1+i)^n - 1}

Donde:
   * C = Cuota mensual
   * M = Monto del crédito
   * i = Tasa mensual (tasa_anual/12/100)
   * n = Número de cuotas (plazo_años × 12)

Métodos de Predicción
---------------------

predict_risk
^^^^^^^^^^^^

.. automethod:: src.prediction.CreditRiskPredictor.predict_risk
   :noindex:

Predice el riesgo crediticio del solicitante.

**Parameters:**
   * ``applicant_data`` (Dict): Datos del solicitante con características derivadas

**Returns:**
   Diccionario con resultados de predicción

**Resultados incluyen:**

* ``predicted_class``: Clase predicha (Bajo/Medio/Alto)
* ``probabilities``: Probabilidades por clase
* ``explanation``: Explicación en lenguaje natural
* ``recommendation``: Recomendación de aprobación
* ``risk_factors``: Factores de riesgo identificados
* ``applicant_data``: Datos completos del solicitante

**Ejemplo:**

.. code-block:: python

   # Predecir
   results = predictor.predict_risk(enhanced_data)
   
   # Resultado principal
   print(f"Riesgo: {results['predicted_class']}")
   print(f"Confianza: {max(results['probabilities'].values()):.1%}")
   
   # Probabilidades
   for clase, prob in results['probabilities'].items():
       print(f"  {clase}: {prob:.1%}")
   
   # Recomendación
   rec = results['recommendation']
   print(f"\nRecomendación: {rec['decision']}")
   print(f"Confianza: {rec['confidence']:.1%}")
   
   # Factores de riesgo
   print("\nFactores de riesgo:")
   for factor in results['risk_factors']:
       print(f"  {factor['factor']}: {factor['value']} - Impacto {factor['impact']}")

Métodos de Explicación
-----------------------

_generate_explanation
^^^^^^^^^^^^^^^^^^^^^

Genera explicación en lenguaje natural de la predicción.

**Factores considerados:**

* Puntaje DataCrédito (< 600 o > 750)
* DTI (> 35% o < 25%)
* Capacidad residual (< 0 o > 500,000)
* Estabilidad laboral (Formal + antigüedad > 3 años)
* Tipo de empleo (Informal)

**Ejemplo de explicación:**

   "El solicitante presenta riesgo **BAJO** con 85.3% de confianza. Esto se debe principalmente a: excelente puntaje DataCrédito (780), bajo ratio de endeudamiento (22.5%), buena capacidad residual, empleo formal estable."

_generate_recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^

Genera recomendación de aprobación del crédito.

**Decisiones posibles:**

1. **APROBAR:** Riesgo Bajo con confianza > 70%
2. **RECHAZAR:** Riesgo Alto o confianza > 80%
3. **REVISAR MANUALMENTE:** Casos intermedios

**Condiciones adicionales evaluadas:**
   * DTI > 40%
   * Número de demandas > 0
   * Capacidad residual < 0

_identify_risk_factors
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.prediction.CreditRiskPredictor._identify_risk_factors
   :noindex:

Identifica los principales factores de riesgo.

**Returns:**
   Lista de diccionarios con factores

**Factores analizados:**

1. **Puntaje DataCrédito**
2. **Ratio Deuda/Ingreso (DTI)**
3. **Capacidad Residual**
4. **Loan-to-Value (LTV)**
5. **Estabilidad Laboral**

**Niveles de impacto:**
   * ALTO: Aumenta significativamente el riesgo
   * MEDIO: Impacto neutral
   * BAJO: Disminuye el riesgo

Funciones de Renderizado
-------------------------

render_prediction_interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.prediction.render_prediction_interface

Renderiza la interfaz completa de predicción en Streamlit.

**Funcionalidades:**
   * Selección de modelo
   * Formulario interactivo de datos
   * Validaciones en tiempo real
   * Predicción con visualizaciones
   * Análisis de factores de riesgo
   * Recomendación de aprobación
   * Historial de predicciones

render_prediction_module
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.prediction.render_prediction_module

Función principal para renderizar el módulo de predicción.

Funciones Auxiliares
--------------------

_save_prediction_to_history
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.prediction._save_prediction_to_history

Guarda la predicción en el historial JSON.

**Archivo:** ``data/predictions_history.json``

**Estructura del historial:**

.. code-block:: python

   [
       {
           'timestamp': '2024-01-01T12:00:00',
           'model_used': 'xgboost',
           'prediction': 'Bajo',
           'probabilities': {'Bajo': 0.85, 'Medio': 0.12, 'Alto': 0.03},
           'recommendation': 'APROBAR',
           'applicant_summary': {
               'edad': 35,
               'salario': 4500000,
               'puntaje_datacredito': 720,
               'dti': 22.5
           }
       }
   ]

Validaciones en Tiempo Real
----------------------------

El sistema realiza validaciones automáticas:

1. **DTI estimado:** Alerta si > 40%
2. **Capacidad de ahorro:** Error si ≤ 0
3. **Consistencia monto:** Error si monto > valor inmueble
4. **Edad + Plazo:** Advertencia si > 75 años
5. **Cuota inicial:** Validación de porcentaje (10-50%)

Ejemplo Completo de Predicción
-------------------------------

.. code-block:: python

   from src.prediction import CreditRiskPredictor
   
   # Inicializar predictor
   predictor = CreditRiskPredictor()
   
   # Cargar modelo
   predictor.load_model('xgboost')
   
   # Datos del solicitante
   applicant = {
       # Personal
       'edad': 38,
       'ciudad': 'Bogotá',
       'estrato_socioeconomico': 4,
       'nivel_educacion': 'Profesional',
       'estado_civil': 'Casado',
       'personas_a_cargo': 2,
       
       # Laboral
       'tipo_empleo': 'Formal',
       'antiguedad_empleo': 6.5,
       'salario_mensual': 5500000,
       'egresos_mensuales': 3200000,
       
       # Financiero
       'puntaje_datacredito': 750,
       'patrimonio_total': 85000000,
       'numero_propiedades': 1,
       'saldo_promedio_banco': 12000000,
       'numero_demandas': 0,
       
       # Inmueble
       'valor_inmueble': 220000000,
       'porcentaje_cuota_inicial': 25,
       'anos_inmueble': 3,
       'plazo_credito': 20,
       'tasa_interes_anual': 10.8
   }
   
   # Calcular características derivadas
   enhanced_data = predictor.calculate_derived_features(applicant)
   
   print("Características derivadas:")
   print(f"  Monto crédito: ${enhanced_data['monto_credito']:,.0f}")
   print(f"  Cuota mensual: ${enhanced_data['cuota_mensual']:,.0f}")
   print(f"  DTI: {enhanced_data['dti']:.1f}%")
   print(f"  LTV: {enhanced_data['ltv']:.1f}%")
   print(f"  Capacidad residual: ${enhanced_data['capacidad_residual']:,.0f}")
   
   # Realizar predicción
   results = predictor.predict_risk(enhanced_data)
   
   # Mostrar resultados
   print(f"\n{'='*60}")
   print(f"RESULTADO DE LA PREDICCIÓN")
   print(f"{'='*60}")
   
   print(f"\n🎯 Riesgo Predicho: {results['predicted_class']}")
   print(f"   Confianza: {max(results['probabilities'].values()):.1%}")
   
   print(f"\nProbabilidades por clase:")
   for clase, prob in results['probabilities'].items():
       emoji = '🟢' if clase == 'Bajo' else '🟡' if clase == 'Medio' else '🔴'
       print(f"  {emoji} {clase}: {prob:.1%}")
   
   print(f"\n💼 Recomendación: {results['recommendation']['decision']}")
   
   if results['recommendation']['conditions']:
       print(f"\nCondiciones a considerar:")
       for condition in results['recommendation']['conditions']:
           print(f"  - {condition}")
   
   print(f"\n📝 Explicación:")
   print(f"   {results['explanation']}")
   
   print(f"\n⚠️ Factores de Riesgo:")
   for factor in results['risk_factors']:
       emoji = '🔴' if factor['impact'] == 'ALTO' else '🟡' if factor['impact'] == 'MEDIO' else '🟢'
       print(f"  {emoji} {factor['factor']}: {factor['value']} ({factor['direction']} riesgo)")

Métodos de Cálculo
------------------

calculate_derived_features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.prediction.CreditRiskPredictor.calculate_derived_features
   :noindex:

Calcula todas las características derivadas necesarias para la predicción.

**Características calculadas:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Característica
     - Fórmula
   * - valor_cuota_inicial
     - valor_inmueble × (porcentaje_cuota_inicial / 100)
   * - monto_credito
     - valor_inmueble - valor_cuota_inicial
   * - cuota_mensual
     - Fórmula de amortización francesa
   * - ltv
     - (monto_credito / valor_inmueble) × 100
   * - dti
     - (cuota_mensual / salario_mensual) × 100
   * - capacidad_ahorro
     - salario_mensual - egresos_mensuales
   * - capacidad_residual
     - capacidad_ahorro - cuota_mensual

predict_risk
^^^^^^^^^^^^

.. automethod:: src.prediction.CreditRiskPredictor.predict_risk
   :noindex:

Realiza la predicción de riesgo crediticio.

**Proceso:**

1. Preparar características según modelo
2. Escalar datos
3. Realizar predicción
4. Calcular probabilidades
5. Generar explicación
6. Generar recomendación
7. Identificar factores de riesgo

Métodos de Generación de Insights
----------------------------------

_generate_explanation
^^^^^^^^^^^^^^^^^^^^^

Genera explicación en lenguaje natural.

**Factores analizados:**

* Puntaje DataCrédito (rangos: <600, >750)
* DTI (rangos: >35%, <25%)
* Capacidad residual (rangos: <0, >500,000)
* Estabilidad laboral (Formal + antigüedad >3)
* Tipo de empleo (Informal)

_generate_recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^

Genera recomendación de aprobación.

**Lógica de decisión:**

.. code-block:: python

   if riesgo == 'Bajo' and confianza > 0.7:
       decision = "APROBAR"
   elif riesgo == 'Alto' or confianza > 0.8:
       decision = "RECHAZAR"
   else:
       decision = "REVISAR MANUALMENTE"

_identify_risk_factors
^^^^^^^^^^^^^^^^^^^^^^

Identifica y clasifica factores de riesgo.

**Clasificación de impacto:**

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Factor
     - Impacto ALTO
     - Impacto BAJO
   * - Puntaje DataCrédito
     - < 600
     - > 750
   * - DTI
     - > 35%
     - < 25%
   * - Capacidad Residual
     - < 0
     - > 500,000
   * - LTV
     - > 85%
     - < 70%
   * - Estabilidad Laboral
     - Informal o <1 año
     - Formal + >3 años

Funciones de Renderizado
-------------------------

render_prediction_interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.prediction.render_prediction_interface

Renderiza la interfaz completa de predicción en Streamlit.

**Componentes de la interfaz:**

1. Selección de modelo con métricas
2. Formulario interactivo en tabs
3. Validaciones en tiempo real
4. Botón de predicción
5. Visualización de resultados:
   
   * Predicción principal con color
   * Probabilidades por clase
   * Gráfico de barras de probabilidades
   * Análisis de factores de riesgo
   * Recomendación final
   * Explicación detallada

render_prediction_module
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.prediction.render_prediction_module

Función principal para renderizar el módulo.

Funciones Auxiliares
--------------------

_save_prediction_to_history
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.prediction._save_prediction_to_history

Guarda predicción en historial JSON.

**Parameters:**
   * ``predictor`` (CreditRiskPredictor): Instancia del predictor
   * ``prediction_results`` (Dict): Resultados de la predicción

**Archivo de historial:**
   * Ubicación: ``data/predictions_history.json``
   * Límite: Últimas 100 predicciones
   * Formato: JSON con timestamp

Ejemplo de Batch Prediction
----------------------------

.. code-block:: python

   from src.prediction import CreditRiskPredictor
   import pandas as pd
   
   # Cargar predictor
   predictor = CreditRiskPredictor()
   predictor.load_model('xgboost')
   
   # Cargar múltiples solicitantes
   applicants_df = pd.read_csv("nuevos_solicitantes.csv")
   
   # Predecir para cada uno
   predictions = []
   
   for idx, row in applicants_df.iterrows():
       # Convertir fila a diccionario
       applicant_data = row.to_dict()
       
       # Calcular características
       enhanced_data = predictor.calculate_derived_features(applicant_data)
       
       # Predecir
       result = predictor.predict_risk(enhanced_data)
       
       predictions.append({
           'id': idx,
           'riesgo_predicho': result['predicted_class'],
           'confianza': max(result['probabilities'].values()),
           'recomendacion': result['recommendation']['decision'],
           'prob_bajo': result['probabilities'].get('Bajo', 0),
           'prob_medio': result['probabilities'].get('Medio', 0),
           'prob_alto': result['probabilities'].get('Alto', 0)
       })
   
   # Crear DataFrame de resultados
   results_df = pd.DataFrame(predictions)
   
   # Guardar resultados
   results_df.to_csv("predicciones_batch.csv", index=False)
   
   # Resumen
   print(f"Total predicciones: {len(results_df)}")
   print(f"\nDistribución de riesgo:")
   print(results_df['riesgo_predicho'].value_counts())
   print(f"\nRecomendaciones:")
   print(results_df['recomendacion'].value_counts())

Integración con Otros Módulos
------------------------------

El módulo de predicción se integra con:

* **feature_engineering:** Para calcular características derivadas
* **supervised_models:** Carga modelos entrenados
* **data_processor:** Validación de datos de entrada

**Flujo completo:**

.. code-block:: python

   # 1. Generar datos
   from src.generar_datos import generar_datos_credito_realista
   df = generar_datos_credito_realista(n_registros=10000)
   
   # 2. Ingeniería de características
   from src.feature_engineering import FeatureEngineer
   engineer = FeatureEngineer(df)
   df_enhanced = engineer.generate_all_features()
   
   # 3. Entrenar modelo
   from src.supervised_models import SupervisedModelTrainer
   trainer = SupervisedModelTrainer()
   trainer.prepare_data(df_enhanced)
   results = trainer.train_model('xgboost')
   
   # 4. Usar para predicción
   from src.prediction import CreditRiskPredictor
   predictor = CreditRiskPredictor()
   predictor.load_model('xgboost')
   
   # 5. Predecir nuevo solicitante
   new_applicant = {...}  # Datos del formulario
   enhanced = predictor.calculate_derived_features(new_applicant)
   prediction = predictor.predict_risk(enhanced)

Ver también
-----------

* :doc:`supervised_models` - Entrenamiento de modelos
* :doc:`feature_engineering` - Características derivadas
* :doc:`retraining` - Reentrenamiento de modelos