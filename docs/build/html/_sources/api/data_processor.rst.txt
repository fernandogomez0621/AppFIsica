data_processor
==============

.. automodule:: src.data_processor
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Módulo de procesamiento, validación y limpieza de datos de crédito hipotecario. Incluye validaciones automáticas, detección de outliers y generación de reportes de calidad.

Clases Principales
------------------

DataProcessor
^^^^^^^^^^^^^

.. autoclass:: src.data_processor.DataProcessor
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Procesador completo de datos con validaciones, limpieza y transformaciones.
   
   **Atributos:**
   
   * ``required_columns``: Columnas requeridas en el dataset
   * ``numeric_ranges``: Rangos válidos para variables numéricas
   * ``categorical_columns``: Lista de columnas categóricas
   * ``validation_report``: Reporte de validación generado
   * ``processing_report``: Reporte de procesamiento
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.data_processor import DataProcessor
      import pandas as pd
      
      # Crear procesador
      processor = DataProcessor()
      
      # Cargar datos
      df = processor.load_data(file_path="datos.csv")
      
      # Validar calidad
      report = processor.validate_data(df)
      
      # Limpiar datos
      df_clean = processor.clean_data(
          df,
          remove_duplicates=True,
          handle_missing='impute',
          handle_outliers='cap'
      )

Métodos Principales
-------------------

load_data
^^^^^^^^^

.. automethod:: src.data_processor.DataProcessor.load_data
   :noindex:

Carga datos desde archivo o upload de Streamlit.

**Parameters:**
   * ``file_path`` (str, optional): Ruta al archivo
   * ``uploaded_file`` (optional): Archivo subido en Streamlit

**Returns:**
   DataFrame cargado o None si hay error

**Formatos soportados:**
   * CSV (.csv)
   * Excel (.xlsx, .xls)
   * Parquet (.parquet)

validate_data
^^^^^^^^^^^^^

.. automethod:: src.data_processor.DataProcessor.validate_data
   :noindex:

Valida la calidad y consistencia de los datos.

**Parameters:**
   * ``df`` (DataFrame): DataFrame a validar

**Returns:**
   Diccionario con reporte de validación

**Validaciones realizadas:**

1. Verificación de columnas requeridas
2. Análisis de valores faltantes
3. Verificación de rangos lógicos
4. Detección de inconsistencias
5. Identificación de duplicados
6. Detección de outliers extremos

clean_data
^^^^^^^^^^

.. automethod:: src.data_processor.DataProcessor.clean_data
   :noindex:

Limpia y procesa los datos según configuración.

**Parameters:**
   * ``df`` (DataFrame): DataFrame a procesar
   * ``remove_duplicates`` (bool): Si eliminar duplicados (default: True)
   * ``handle_missing`` (str): Estrategia para valores faltantes ('drop', 'impute', 'keep')
   * ``handle_outliers`` (str): Estrategia para outliers ('remove', 'cap', 'keep')
   * ``normalize_numeric`` (bool): Si normalizar variables numéricas (default: False)

**Returns:**
   DataFrame procesado

**Estrategias de imputación:**
   * Numéricas: Mediana
   * Categóricas: Moda

create_quality_report_visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.data_processor.DataProcessor.create_quality_report_visualizations
   :noindex:

Crea visualizaciones del reporte de calidad.

**Parameters:**
   * ``df`` (DataFrame): DataFrame a analizar

**Returns:**
   Diccionario con figuras de Plotly

**Visualizaciones generadas:**
   * Heatmap de valores faltantes
   * Gráfico de barras de valores faltantes
   * Distribución de tipos de datos

Funciones de Renderizado
-------------------------

render_data_loader
^^^^^^^^^^^^^^^^^^

.. autofunction:: src.data_processor.render_data_loader

Renderiza el módulo de carga de datos en Streamlit.

**Funcionalidades:**
   * Carga de archivos (CSV, Excel, Parquet)
   * Vista previa de datos
   * Validación automática de calidad
   * Procesamiento interactivo
   * Descarga de datos procesados

render_data_processor_module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.data_processor.render_data_processor_module

Función principal para renderizar el módulo completo de procesamiento.

Rangos de Validación
---------------------

El procesador valida los siguientes rangos:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20

   * - Variable
     - Mínimo
     - Máximo
   * - edad
     - 18
     - 80
   * - salario_mensual
     - 1,000,000
     - 50,000,000
   * - puntaje_datacredito
     - 150
     - 950
   * - valor_inmueble
     - 20,000,000
     - 2,000,000,000
   * - tasa_interes_anual
     - 5.0
     - 25.0
   * - plazo_credito
     - 5
     - 30
   * - dti
     - 0
     - 60
   * - ltv
     - 0
     - 100

Estructura del Reporte de Validación
-------------------------------------

El reporte de validación incluye:

.. code-block:: python

   {
       'timestamp': '2024-01-01T12:00:00',
       'total_rows': 10000,
       'total_columns': 45,
       'warnings': [
           'Columnas con >20% valores faltantes: [...]',
           'Outliers detectados en: [...]'
       ],
       'errors': [
           'Columnas faltantes: [...]',
           'Inconsistencias: monto_credito > valor_inmueble'
       ],
       'suggestions': [
           '✅ Todas las columnas requeridas presentes'
       ]
   }

Ejemplo Completo
----------------

.. code-block:: python

   from src.data_processor import DataProcessor
   import pandas as pd
   
   # Inicializar procesador
   processor = DataProcessor()
   
   # Cargar datos
   df = pd.read_csv("datos_credito.csv")
   
   # Validar
   validation_report = processor.validate_data(df)
   print(f"Errores: {len(validation_report['errors'])}")
   print(f"Advertencias: {len(validation_report['warnings'])}")
   
   # Limpiar
   df_clean = processor.clean_data(
       df,
       remove_duplicates=True,
       handle_missing='impute',
       handle_outliers='cap',
       normalize_numeric=False
   )
   
   # Crear visualizaciones
   figures = processor.create_quality_report_visualizations(df_clean)
   
   # Guardar datos limpios
   df_clean.to_csv("datos_procesados.csv", index=False)
   
   print(f"Procesamiento completado: {len(df_clean)} registros")

Ver también
-----------

* :doc:`generar_datos` - Generación de datos sintéticos
* :doc:`univariate_analysis` - Análisis univariado
* :doc:`feature_engineering` - Ingeniería de características