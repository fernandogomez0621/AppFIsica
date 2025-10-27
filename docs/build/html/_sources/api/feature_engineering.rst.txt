feature_engineering
===================

.. automodule:: src.feature_engineering
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Módulo de ingeniería de características para crear variables derivadas automáticamente. Mejora el poder predictivo de los modelos mediante transformaciones, ratios e interacciones.

Clases Principales
------------------

FeatureEngineer
^^^^^^^^^^^^^^^

.. autoclass:: src.feature_engineering.FeatureEngineer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Ingeniero de características que crea variables derivadas automáticamente.
   
   **Parámetros del constructor:**
   
   :param data: DataFrame con datos originales
   :type data: pd.DataFrame
   
   **Atributos:**
   
   * ``data``: DataFrame original
   * ``original_columns``: Lista de columnas originales
   * ``new_features``: Diccionario de nuevas características creadas
   * ``feature_importance``: Importancia de características
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.feature_engineering import FeatureEngineer
      import pandas as pd
      
      # Cargar datos
      df = pd.read_csv("datos_credito.csv")
      
      # Crear ingeniero
      engineer = FeatureEngineer(df)
      
      # Generar todas las características
      df_enhanced = engineer.generate_all_features()
      
      # Calcular importancia
      importance = engineer.calculate_feature_importance(df_enhanced)

Métodos de Creación de Características
---------------------------------------

create_financial_ratios
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.feature_engineering.FeatureEngineer.create_financial_ratios
   :noindex:

Crea ratios financieros fundamentales.

**Returns:**
   DataFrame con ratios financieros

**Ratios creados:**

1. **ltv_ratio:** Loan-to-Value (Préstamo/Valor inmueble) %
2. **dti_ratio:** Debt-to-Income (Deuda/Ingreso) %
3. **capacidad_ahorro_nueva:** Salario - Egresos
4. **ratio_ahorro_salario:** (Ahorro/Salario) %
5. **ratio_patrimonio_deuda:** Patrimonio/Deuda
6. **saldo_relativo:** Saldo banco/Salario
7. **meses_colchon:** Saldo/Cuota mensual
8. **ratio_cuota_inicial:** (Cuota inicial/Valor inmueble) %

**Ejemplo:**

.. code-block:: python

   df_with_ratios = engineer.create_financial_ratios()
   
   # Ver nuevos ratios
   print(df_with_ratios[['ltv_ratio', 'dti_ratio', 'meses_colchon']].head())

create_risk_indicators
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.feature_engineering.FeatureEngineer.create_risk_indicators
   :noindex:

Crea indicadores de riesgo específicos.

**Parameters:**
   * ``df`` (DataFrame): DataFrame con ratios financieros

**Returns:**
   DataFrame con indicadores de riesgo

**Indicadores creados:**

1. **score_edad:** Penalización por edades extremas
2. **flag_sobreendeudamiento:** DTI > 40% (binario)
3. **nivel_sobreendeudamiento:** Bajo/Moderado/Alto/Crítico
4. **score_estabilidad:** Estabilidad laboral (0-125)
5. **riesgo_legal:** Función exponencial de demandas
6. **score_educacion:** Codificación ordinal (1-4)
7. **flag_alta_liquidez:** Saldo > 3 salarios (binario)

**Ejemplo:**

.. code-block:: python

   df_with_indicators = engineer.create_risk_indicators(df)
   
   # Ver indicadores
   print(df_with_indicators[['score_edad', 'score_estabilidad', 'riesgo_legal']].head())

create_interaction_features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.feature_engineering.FeatureEngineer.create_interaction_features
   :noindex:

Crea variables de interacción entre características.

**Parameters:**
   * ``df`` (DataFrame): DataFrame con características base

**Returns:**
   DataFrame con interacciones

**Interacciones creadas:**

1. **educacion_x_salario:** Educación × Salario/1M
2. **propiedades_x_patrimonio:** Propiedades × log(Patrimonio)
3. **edad_x_empleo:** Edad × Antigüedad empleo
4. **ltv_x_puntaje:** LTV × (900 - Puntaje DataCrédito)/100

**Ejemplo:**

.. code-block:: python

   df_with_interactions = engineer.create_interaction_features(df)
   
   # Ver interacciones
   print(df_with_interactions[['educacion_x_salario', 'ltv_x_puntaje']].head())

create_binned_features
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.feature_engineering.FeatureEngineer.create_binned_features
   :noindex:

Crea variables discretizadas/categorizadas.

**Parameters:**
   * ``df`` (DataFrame): DataFrame con variables continuas

**Returns:**
   DataFrame con variables discretizadas

**Variables discretizadas:**

1. **grupo_edad:** Joven/Adulto_Joven/Adulto/Adulto_Mayor
2. **rango_salarial:** Muy_Bajo/Bajo/Medio/Alto/Muy_Alto
3. **categoria_puntaje:** Malo/Regular/Bueno/Muy_Bueno/Excelente
4. **nivel_ltv:** Muy_Bajo/Bajo/Medio/Alto/Muy_Alto

**Ejemplo:**

.. code-block:: python

   df_binned = engineer.create_binned_features(df)
   
   # Ver distribución de grupos
   print(df_binned['grupo_edad'].value_counts())
   print(df_binned['rango_salarial'].value_counts())

create_transformed_features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.feature_engineering.FeatureEngineer.create_transformed_features
   :noindex:

Crea variables transformadas matemáticamente.

**Parameters:**
   * ``df`` (DataFrame): DataFrame con variables originales

**Returns:**
   DataFrame con transformaciones

**Transformaciones aplicadas:**

* **Logarítmica:** Para variables con distribución sesgada
* **Raíz cuadrada:** Alternativa a log para valores pequeños
* **Cuadrática:** Para capturar relaciones no lineales

**Variables transformadas:**
   * salario_mensual_log, salario_mensual_sqrt
   * patrimonio_total_log, patrimonio_total_sqrt
   * valor_inmueble_log, valor_inmueble_sqrt
   * saldo_promedio_banco_log, saldo_promedio_banco_sqrt
   * dti_cuadrado
   * edad_cuadrado

calculate_feature_importance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.feature_engineering.FeatureEngineer.calculate_feature_importance
   :noindex:

Calcula importancia de características usando mutual information.

**Parameters:**
   * ``df`` (DataFrame): DataFrame con características
   * ``target_col`` (str): Variable objetivo (default: 'nivel_riesgo')

**Returns:**
   Diccionario ordenado con importancias

**Ejemplo:**

.. code-block:: python

   importance = engineer.calculate_feature_importance(df_enhanced, 'nivel_riesgo')
   
   # Top 10 características
   for feature, score in list(importance.items())[:10]:
       print(f"{feature}: {score:.4f}")

generate_all_features
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.feature_engineering.FeatureEngineer.generate_all_features
   :noindex:

Genera todas las características derivadas en un solo paso.

**Returns:**
   DataFrame con todas las características

**Proceso completo:**

1. Ratios financieros
2. Indicadores de riesgo
3. Variables de interacción
4. Variables discretizadas
5. Transformaciones matemáticas

Funciones de Renderizado
-------------------------

render_feature_engineering
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.feature_engineering.render_feature_engineering

Renderiza el módulo de ingeniería de características en Streamlit.

render_feature_engineering_module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.feature_engineering.render_feature_engineering_module

Función principal para renderizar el módulo completo.

Ejemplo Completo de Uso
------------------------

.. code-block:: python

   from src.feature_engineering import FeatureEngineer
   import pandas as pd
   
   # Cargar datos originales
   df = pd.read_csv("datos_credito_hipotecario_realista.csv")
   print(f"Características originales: {len(df.columns)}")
   
   # Crear ingeniero
   engineer = FeatureEngineer(df)
   
   # Generar todas las características
   df_enhanced = engineer.generate_all_features()
   print(f"Características totales: {len(df_enhanced.columns)}")
   print(f"Nuevas características: {len(engineer.new_features)}")
   
   # Ver nuevas características
   print("\nNuevas características creadas:")
   for feature, description in engineer.new_features.items():
       print(f"  - {feature}: {description}")
   
   # Calcular importancia
   importance = engineer.calculate_feature_importance(df_enhanced)
   
   print("\nTop 10 características más importantes:")
   for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
       print(f"{i:2d}. {feature:30s}: {score:.4f}")
   
   # Guardar dataset enriquecido
   df_enhanced.to_csv("datos_con_caracteristicas.csv", index=False)
   print(f"\n✅ Dataset enriquecido guardado")

Fórmulas de Características
----------------------------

Ratios Financieros
^^^^^^^^^^^^^^^^^^

.. math::

   LTV = \frac{\text{Monto Crédito}}{\text{Valor Inmueble}} \times 100

.. math::

   DTI = \frac{\text{Cuota Mensual}}{\text{Salario Mensual}} \times 100

.. math::

   \text{Capacidad Ahorro} = \text{Salario} - \text{Egresos}

.. math::

   \text{Meses Colchón} = \frac{\text{Saldo Banco}}{\text{Cuota Mensual}}

Indicadores de Riesgo
^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{Riesgo Legal} = 100 \times (1 - e^{-2 \times \text{Demandas}})

.. math::

   \text{Score Estabilidad} = \min(125, \text{Antigüedad} \times 10 + \text{Bonus Empleo})

Ver también
-----------

* :doc:`data_processor` - Procesamiento de datos
* :doc:`supervised_models` - Modelos de clasificación
* :doc:`rbm_model` - Características RBM