univariate_analysis
===================

.. automodule:: src.univariate_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Módulo de análisis estadístico descriptivo univariado. Proporciona estadísticas completas y visualizaciones interactivas para variables individuales.

Clases Principales
------------------

UnivariateAnalyzer
^^^^^^^^^^^^^^^^^^

.. autoclass:: src.univariate_analysis.UnivariateAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Analizador de estadísticas univariadas para variables numéricas y categóricas.
   
   **Parámetros del constructor:**
   
   :param data: DataFrame con los datos a analizar
   :type data: pd.DataFrame
   
   **Atributos:**
   
   * ``data``: DataFrame con los datos
   * ``numeric_columns``: Lista de columnas numéricas
   * ``categorical_columns``: Lista de columnas categóricas
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.univariate_analysis import UnivariateAnalyzer
      import pandas as pd
      
      # Cargar datos
      df = pd.read_csv("datos_credito.csv")
      
      # Crear analizador
      analyzer = UnivariateAnalyzer(df)
      
      # Analizar variable numérica
      stats = analyzer.analyze_numeric_variable('salario_mensual')
      print(f"Media: {stats['mean']:,.2f}")
      print(f"Mediana: {stats['median']:,.2f}")
      
      # Crear visualizaciones
      figures = analyzer.create_numeric_visualizations('salario_mensual')

Métodos de Análisis
-------------------

analyze_numeric_variable
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.univariate_analysis.UnivariateAnalyzer.analyze_numeric_variable
   :noindex:

Análisis completo de variable numérica.

**Parameters:**
   * ``column`` (str): Nombre de la columna a analizar

**Returns:**
   Diccionario con estadísticas completas

**Estadísticas calculadas:**

* **Tendencia central:** media, mediana, moda
* **Dispersión:** desviación estándar, varianza, rango, IQR
* **Forma:** asimetría (skewness), curtosis
* **Percentiles:** P5, Q1, Q3, P95
* **Normalidad:** Test de Shapiro-Wilk o Kolmogorov-Smirnov
* **Outliers:** Detección mediante método IQR

**Ejemplo:**

.. code-block:: python

   stats = analyzer.analyze_numeric_variable('edad')
   
   print(f"Media: {stats['mean']:.2f}")
   print(f"Desviación estándar: {stats['std']:.2f}")
   print(f"Asimetría: {stats['skewness']:.3f}")
   print(f"Es normal: {stats['is_normal']}")
   print(f"Outliers: {stats['outliers_count']} ({stats['outliers_pct']:.1f}%)")

analyze_categorical_variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.univariate_analysis.UnivariateAnalyzer.analyze_categorical_variable
   :noindex:

Análisis completo de variable categórica.

**Parameters:**
   * ``column`` (str): Nombre de la columna a analizar

**Returns:**
   Diccionario con estadísticas

**Estadísticas calculadas:**

* **Conteo:** total de observaciones, valores únicos
* **Moda:** categoría más frecuente y su porcentaje
* **Distribución:** frecuencias absolutas y relativas
* **Entropía:** Medida de diversidad de categorías

**Ejemplo:**

.. code-block:: python

   stats = analyzer.analyze_categorical_variable('nivel_riesgo')
   
   print(f"Valores únicos: {stats['unique_values']}")
   print(f"Moda: {stats['mode']} ({stats['mode_percentage']:.1f}%)")
   print(f"Entropía: {stats['entropy']:.3f}")
   
   # Ver distribución
   for categoria, frecuencia in stats['value_counts'].items():
       porcentaje = stats['frequencies'][categoria]
       print(f"{categoria}: {frecuencia} ({porcentaje:.1f}%)")

Métodos de Visualización
-------------------------

create_numeric_visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.univariate_analysis.UnivariateAnalyzer.create_numeric_visualizations
   :noindex:

Crea visualizaciones para variable numérica.

**Parameters:**
   * ``column`` (str): Nombre de la columna

**Returns:**
   Diccionario con figuras de Plotly

**Visualizaciones generadas:**

1. **Histograma con KDE:** Distribución de frecuencias con curva de densidad
2. **Boxplot:** Diagrama de caja con outliers
3. **Q-Q Plot:** Evaluación de normalidad
4. **ECDF:** Función de distribución empírica acumulada

**Ejemplo:**

.. code-block:: python

   figures = analyzer.create_numeric_visualizations('puntaje_datacredito')
   
   # Mostrar histograma
   figures['histogram'].show()
   
   # Mostrar boxplot
   figures['boxplot'].show()

create_categorical_visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.univariate_analysis.UnivariateAnalyzer.create_categorical_visualizations
   :noindex:

Crea visualizaciones para variable categórica.

**Parameters:**
   * ``column`` (str): Nombre de la columna

**Returns:**
   Diccionario con figuras de Plotly

**Visualizaciones generadas:**

1. **Gráfico de barras:** Frecuencias por categoría
2. **Gráfico de torta:** Distribución porcentual

Funciones de Renderizado
-------------------------

render_univariate_analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.univariate_analysis.render_univariate_analysis

Renderiza el módulo de análisis univariado en Streamlit.

**Funcionalidades:**
   * Selector interactivo de variables
   * Análisis automático según tipo de variable
   * Visualizaciones interactivas
   * Interpretaciones estadísticas

render_numeric_analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.univariate_analysis.render_numeric_analysis

Renderiza análisis específico para variable numérica.

**Parameters:**
   * ``analyzer`` (UnivariateAnalyzer): Analizador inicializado
   * ``column`` (str): Nombre de la columna
   * ``df`` (DataFrame): DataFrame con datos

render_categorical_analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.univariate_analysis.render_categorical_analysis

Renderiza análisis específico para variable categórica.

**Parameters:**
   * ``analyzer`` (UnivariateAnalyzer): Analizador inicializado
   * ``column`` (str): Nombre de la columna
   * ``df`` (DataFrame): DataFrame con datos

render_univariate_module
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.univariate_analysis.render_univariate_module

Función principal para renderizar el módulo univariado completo.

Tests Estadísticos
------------------

Test de Normalidad
^^^^^^^^^^^^^^^^^^

El módulo implementa dos tests de normalidad:

* **Shapiro-Wilk:** Para n < 5000 muestras
* **Kolmogorov-Smirnov:** Para n ≥ 5000 muestras

**Interpretación:**
   * p-value > 0.05: Distribución normal
   * p-value ≤ 0.05: No sigue distribución normal

Detección de Outliers
^^^^^^^^^^^^^^^^^^^^^^

Método IQR (Rango Intercuartílico):

.. math::

   \text{Límite inferior} = Q1 - 1.5 \times IQR
   
   \text{Límite superior} = Q3 + 1.5 \times IQR

Donde:
   * Q1 = Cuartil 1 (percentil 25)
   * Q3 = Cuartil 3 (percentil 75)
   * IQR = Q3 - Q1

Interpretación de Estadísticas
-------------------------------

Asimetría (Skewness)
^^^^^^^^^^^^^^^^^^^^

* **< -1:** Asimetría negativa fuerte (cola izquierda)
* **-1 a -0.5:** Asimetría negativa moderada
* **-0.5 a 0.5:** Aproximadamente simétrica
* **0.5 a 1:** Asimetría positiva moderada
* **> 1:** Asimetría positiva fuerte (cola derecha)

Curtosis
^^^^^^^^

* **< 0:** Platicúrtica (más plana que normal)
* **≈ 0:** Mesocúrtica (similar a normal)
* **> 0:** Leptocúrtica (más puntiaguda que normal)

Coeficiente de Variación
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   CV = \frac{\sigma}{\mu} \times 100

* **< 15%:** Baja variabilidad
* **15-30%:** Variabilidad moderada
* **> 30%:** Alta variabilidad

Ejemplo Completo de Análisis
-----------------------------

.. code-block:: python

   from src.univariate_analysis import UnivariateAnalyzer
   import pandas as pd
   
   # Cargar datos
   df = pd.read_csv("datos_credito_hipotecario_realista.csv")
   
   # Crear analizador
   analyzer = UnivariateAnalyzer(df)
   
   # Analizar salario
   stats_salario = analyzer.analyze_numeric_variable('salario_mensual')
   
   print("=== ANÁLISIS DE SALARIO MENSUAL ===")
   print(f"Observaciones: {stats_salario['count']:,}")
   print(f"Media: ${stats_salario['mean']:,.0f}")
   print(f"Mediana: ${stats_salario['median']:,.0f}")
   print(f"Desv. Estándar: ${stats_salario['std']:,.0f}")
   print(f"Coef. Variación: {stats_salario['cv']:.1f}%")
   print(f"Asimetría: {stats_salario['skewness']:.3f}")
   print(f"Outliers: {stats_salario['outliers_count']} ({stats_salario['outliers_pct']:.1f}%)")
   
   # Crear visualizaciones
   figures = analyzer.create_numeric_visualizations('salario_mensual')
   figures['histogram'].write_html("histograma_salario.html")
   
   # Analizar nivel de riesgo
   stats_riesgo = analyzer.analyze_categorical_variable('nivel_riesgo')
   
   print("\n=== ANÁLISIS DE NIVEL DE RIESGO ===")
   print(f"Categorías: {stats_riesgo['unique_values']}")
   print(f"Moda: {stats_riesgo['mode']} ({stats_riesgo['mode_percentage']:.1f}%)")
   print("\nDistribución:")
   for cat, pct in stats_riesgo['frequencies'].items():
       print(f"  {cat}: {pct:.1f}%")

Ver también
-----------

* :doc:`bivariate_analysis` - Análisis bivariado
* :doc:`data_processor` - Procesamiento de datos
* :doc:`feature_engineering` - Ingeniería de características