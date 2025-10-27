bivariate_analysis
==================

.. automodule:: src.bivariate_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Módulo de análisis bivariado para estudiar relaciones entre pares de variables. Incluye correlaciones, tablas de contingencia y tests estadísticos.

Clases Principales
------------------

BivariateAnalyzer
^^^^^^^^^^^^^^^^^

.. autoclass:: src.bivariate_analysis.BivariateAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Analizador de relaciones bivariadas entre variables numéricas y categóricas.
   
   **Parámetros del constructor:**
   
   :param data: DataFrame con los datos a analizar
   :type data: pd.DataFrame
   
   **Atributos:**
   
   * ``data``: DataFrame con los datos
   * ``numeric_columns``: Lista de columnas numéricas
   * ``categorical_columns``: Lista de columnas categóricas
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.bivariate_analysis import BivariateAnalyzer
      import pandas as pd
      
      # Cargar datos
      df = pd.read_csv("datos_credito.csv")
      
      # Crear analizador
      analyzer = BivariateAnalyzer(df)
      
      # Analizar correlación
      results = analyzer.analyze_numeric_vs_numeric('edad', 'salario_mensual')
      print(f"Correlación Pearson: {results['correlations']['pearson']['r']:.3f}")

Métodos de Análisis
-------------------

analyze_numeric_vs_numeric
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.bivariate_analysis.BivariateAnalyzer.analyze_numeric_vs_numeric
   :noindex:

Análisis de relación entre dos variables numéricas.

**Parameters:**
   * ``var1`` (str): Primera variable numérica
   * ``var2`` (str): Segunda variable numérica

**Returns:**
   Diccionario con correlaciones y regresión lineal

**Análisis incluidos:**

* **Correlación de Pearson:** Relación lineal
* **Correlación de Spearman:** Relación monotónica
* **Correlación de Kendall:** Relación ordinal
* **Regresión lineal:** Pendiente, intercepto, R², p-valor

**Ejemplo:**

.. code-block:: python

   results = analyzer.analyze_numeric_vs_numeric('dti', 'puntaje_datacredito')
   
   # Correlaciones
   pearson_r = results['correlations']['pearson']['r']
   pearson_p = results['correlations']['pearson']['p_value']
   
   print(f"Correlación Pearson: r={pearson_r:.3f}, p={pearson_p:.6f}")
   
   # Regresión
   slope = results['linear_regression']['slope']
   r_squared = results['linear_regression']['r_squared']
   
   print(f"Pendiente: {slope:.4f}")
   print(f"R²: {r_squared:.4f}")

analyze_categorical_vs_categorical
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.bivariate_analysis.BivariateAnalyzer.analyze_categorical_vs_categorical
   :noindex:

Análisis de relación entre dos variables categóricas.

**Parameters:**
   * ``var1`` (str): Primera variable categórica
   * ``var2`` (str): Segunda variable categórica

**Returns:**
   Diccionario con tabla de contingencia y tests

**Análisis incluidos:**

* **Tabla de contingencia:** Frecuencias cruzadas
* **Test Chi-cuadrado:** Independencia estadística
* **V de Cramér:** Fuerza de asociación (0-1)

**Interpretación V de Cramér:**
   * < 0.1: Asociación muy débil
   * 0.1-0.3: Asociación débil
   * 0.3-0.5: Asociación moderada
   * > 0.5: Asociación fuerte

**Ejemplo:**

.. code-block:: python

   results = analyzer.analyze_categorical_vs_categorical(
       'nivel_educacion', 
       'nivel_riesgo'
   )
   
   # Test Chi-cuadrado
   chi2 = results['chi2_test']['statistic']
   p_value = results['chi2_test']['p_value']
   
   print(f"Chi²: {chi2:.3f}, p-value: {p_value:.6f}")
   
   if p_value < 0.05:
       print("✅ Existe asociación significativa")
   
   # V de Cramér
   cramers_v = results['cramers_v']
   strength = results['association_strength']
   
   print(f"V de Cramér: {cramers_v:.3f} ({strength})")

analyze_numeric_vs_categorical
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.bivariate_analysis.BivariateAnalyzer.analyze_numeric_vs_categorical
   :noindex:

Análisis de relación entre variable numérica y categórica.

**Parameters:**
   * ``numeric_var`` (str): Variable numérica
   * ``categorical_var`` (str): Variable categórica

**Returns:**
   Diccionario con estadísticas por grupo y tests

**Análisis incluidos:**

* **Estadísticas por grupo:** Media, mediana, desv. estándar por categoría
* **Test ANOVA:** Diferencias entre medias (paramétrico)
* **Test Kruskal-Wallis:** Diferencias entre grupos (no paramétrico)

**Ejemplo:**

.. code-block:: python

   results = analyzer.analyze_numeric_vs_categorical(
       'salario_mensual',
       'nivel_riesgo'
   )
   
   # Estadísticas por grupo
   print(results['grouped_statistics'])
   
   # Test ANOVA
   f_stat = results['anova_test']['f_statistic']
   p_value = results['anova_test']['p_value']
   
   print(f"F-estadístico: {f_stat:.3f}")
   print(f"P-valor: {p_value:.6f}")
   
   if p_value < 0.05:
       print("✅ Diferencias significativas entre grupos")

Métodos de Visualización
-------------------------

create_correlation_matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.bivariate_analysis.BivariateAnalyzer.create_correlation_matrix
   :noindex:

Crea matriz de correlación interactiva.

**Parameters:**
   * ``method`` (str): Método de correlación ('pearson', 'spearman', 'kendall')

**Returns:**
   Figura de Plotly con heatmap

create_scatter_plot
^^^^^^^^^^^^^^^^^^^

.. automethod:: src.bivariate_analysis.BivariateAnalyzer.create_scatter_plot
   :noindex:

Crea gráfico de dispersión con línea de regresión.

**Parameters:**
   * ``var1`` (str): Variable para eje X
   * ``var2`` (str): Variable para eje Y
   * ``color_var`` (str, optional): Variable para colorear puntos

**Returns:**
   Figura de Plotly

create_contingency_heatmap
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.bivariate_analysis.BivariateAnalyzer.create_contingency_heatmap
   :noindex:

Crea heatmap de tabla de contingencia.

**Parameters:**
   * ``var1`` (str): Primera variable categórica
   * ``var2`` (str): Segunda variable categórica

**Returns:**
   Figura de Plotly

create_grouped_boxplot
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.bivariate_analysis.BivariateAnalyzer.create_grouped_boxplot
   :noindex:

Crea boxplot agrupado por categoría.

**Parameters:**
   * ``numeric_var`` (str): Variable numérica
   * ``categorical_var`` (str): Variable categórica

**Returns:**
   Figura de Plotly

Funciones de Renderizado
-------------------------

render_bivariate_analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.bivariate_analysis.render_bivariate_analysis

Renderiza el módulo completo de análisis bivariado en Streamlit.

**Funcionalidades:**
   * Matriz de correlación interactiva
   * Análisis numérica vs numérica
   * Análisis categórica vs categórica
   * Análisis numérica vs categórica
   * Visualizaciones interactivas

render_bivariate_module
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.bivariate_analysis.render_bivariate_module

Función principal para renderizar el módulo bivariado.

Ejemplo Completo
----------------

.. code-block:: python

   from src.bivariate_analysis import BivariateAnalyzer
   import pandas as pd
   
   # Cargar datos
   df = pd.read_csv("datos_credito_hipotecario_realista.csv")
   
   # Crear analizador
   analyzer = BivariateAnalyzer(df)
   
   # 1. Matriz de correlación
   fig_corr = analyzer.create_correlation_matrix(method='pearson')
   fig_corr.show()
   
   # 2. Analizar edad vs salario
   results_num = analyzer.analyze_numeric_vs_numeric('edad', 'salario_mensual')
   print(f"Correlación: {results_num['correlations']['pearson']['r']:.3f}")
   
   # 3. Scatter plot
   fig_scatter = analyzer.create_scatter_plot(
       'edad', 
       'salario_mensual',
       color_var='nivel_riesgo'
   )
   fig_scatter.show()
   
   # 4. Analizar educación vs riesgo
   results_cat = analyzer.analyze_categorical_vs_categorical(
       'nivel_educacion',
       'nivel_riesgo'
   )
   print(f"V de Cramér: {results_cat['cramers_v']:.3f}")
   
   # 5. Analizar salario por tipo de empleo
   results_mixed = analyzer.analyze_numeric_vs_categorical(
       'salario_mensual',
       'tipo_empleo'
   )
   print(f"ANOVA p-value: {results_mixed['anova_test']['p_value']:.6f}")

Ver también
-----------

* :doc:`univariate_analysis` - Análisis univariado
* :doc:`feature_engineering` - Ingeniería de características
* :doc:`clustering` - Análisis de clustering