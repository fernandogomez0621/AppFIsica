clustering
==========

.. automodule:: src.clustering
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Módulo de segmentación de solicitantes en grupos homogéneos usando múltiples algoritmos de clustering con visualizaciones PCA 2D/3D interactivas.

Clases Principales
------------------

ClusterAnalyzer
^^^^^^^^^^^^^^^

.. autoclass:: src.clustering.ClusterAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Analizador de clustering para segmentación de clientes de crédito hipotecario.
   
   **Parámetros del constructor:**
   
   :param data: DataFrame con datos a analizar
   :type data: pd.DataFrame
   
   **Atributos:**
   
   * ``data``: DataFrame con los datos
   * ``numeric_columns``: Columnas numéricas disponibles
   * ``scaler``: StandardScaler para normalización
   * ``pca_2d``: PCA con 2 componentes
   * ``pca_3d``: PCA con 3 componentes
   * ``clustering_results``: Resultados de clustering
   * ``optimal_k``: Número óptimo de clusters
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.clustering import ClusterAnalyzer
      import pandas as pd
      
      # Cargar datos
      df = pd.read_csv("datos_con_caracteristicas.csv")
      
      # Crear analizador
      analyzer = ClusterAnalyzer(df)
      
      # Preparar datos
      features = ['edad', 'salario_mensual', 'puntaje_datacredito', 'dti']
      X_scaled, valid_features = analyzer.prepare_data(features)
      
      # Encontrar k óptimo
      k_results = analyzer.find_optimal_k(X_scaled, max_k=10)
      print(f"K óptimo (Silhouette): {k_results['optimal_k_methods']['silhouette']}")
      
      # Ejecutar clustering
      results = analyzer.perform_clustering(X_scaled, 'kmeans', n_clusters=3)
      print(f"Silhouette Score: {results['metrics']['silhouette_score']:.3f}")

Métodos de Preparación
----------------------

prepare_data
^^^^^^^^^^^^

.. automethod:: src.clustering.ClusterAnalyzer.prepare_data
   :noindex:

Prepara y escala datos para clustering.

**Parameters:**
   * ``selected_features`` (List[str]): Lista de características a usar

**Returns:**
   Tupla (datos_escalados, características_válidas)

**Proceso:**

1. Filtrar características numéricas válidas
2. Imputar valores faltantes con mediana
3. Escalar datos con StandardScaler

find_optimal_k
^^^^^^^^^^^^^^

.. automethod:: src.clustering.ClusterAnalyzer.find_optimal_k
   :noindex:

Encuentra el número óptimo de clusters usando múltiples métodos.

**Parameters:**
   * ``X`` (np.ndarray): Datos escalados
   * ``max_k`` (int): Número máximo de clusters a evaluar (default: 10)

**Returns:**
   Diccionario con métricas por k

**Métodos de evaluación:**

1. **Método del Codo:** Busca el "codo" en la curva de inercia
2. **Silhouette Score:** Maximiza cohesión y separación
3. **Davies-Bouldin Index:** Minimiza ratio intra/inter cluster
4. **Calinski-Harabasz Index:** Maximiza ratio varianza entre/dentro

**Ejemplo:**

.. code-block:: python

   k_results = analyzer.find_optimal_k(X_scaled, max_k=10)
   
   print("K óptimo según diferentes métodos:")
   print(f"  Método del Codo: {k_results['optimal_k_methods']['elbow']}")
   print(f"  Silhouette: {k_results['optimal_k_methods']['silhouette']}")
   print(f"  Davies-Bouldin: {k_results['optimal_k_methods']['davies_bouldin']}")
   
   # Ver métricas por k
   for k, silhouette in zip(k_results['k_range'], k_results['silhouette_scores']):
       print(f"k={k}: Silhouette={silhouette:.3f}")

Métodos de Clustering
---------------------

perform_clustering
^^^^^^^^^^^^^^^^^^

.. automethod:: src.clustering.ClusterAnalyzer.perform_clustering
   :noindex:

Ejecuta algoritmo de clustering seleccionado.

**Parameters:**
   * ``X`` (np.ndarray): Datos escalados
   * ``algorithm`` (str): Algoritmo ('kmeans', 'hierarchical', 'dbscan', 'gmm')
   * ``n_clusters`` (int): Número de clusters
   * ``**kwargs``: Parámetros adicionales del algoritmo

**Returns:**
   Diccionario con resultados del clustering

**Algoritmos soportados:**

1. **K-Means:** Clustering particional basado en centroides
2. **Hierarchical:** Clustering jerárquico aglomerativo
3. **DBSCAN:** Clustering basado en densidad
4. **GMM:** Gaussian Mixture Model (modelo probabilístico)

**Ejemplo:**

.. code-block:: python

   # K-Means
   results_kmeans = analyzer.perform_clustering(X_scaled, 'kmeans', n_clusters=3)
   
   # DBSCAN
   results_dbscan = analyzer.perform_clustering(
       X_scaled, 
       'dbscan', 
       n_clusters=None,
       eps=0.5,
       min_samples=5
   )
   
   # Ver métricas
   print(f"Silhouette: {results_kmeans['metrics']['silhouette_score']:.3f}")
   print(f"Davies-Bouldin: {results_kmeans['metrics']['davies_bouldin_score']:.3f}")
   print(f"Tamaños de clusters: {results_kmeans['cluster_sizes']}")

Métodos de Visualización
-------------------------

create_pca_visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.clustering.ClusterAnalyzer.create_pca_visualizations
   :noindex:

Crea visualizaciones PCA 2D y 3D de los clusters.

**Parameters:**
   * ``X`` (np.ndarray): Datos escalados
   * ``labels`` (np.ndarray): Etiquetas de cluster
   * ``feature_names`` (List[str]): Nombres de características

**Returns:**
   Diccionario con figuras de Plotly

**Visualizaciones:**
   * **pca_2d:** Proyección 2D con varianza explicada
   * **pca_3d:** Proyección 3D interactiva

**Ejemplo:**

.. code-block:: python

   pca_figures = analyzer.create_pca_visualizations(
       X_scaled,
       results['labels'],
       feature_names
   )
   
   # Mostrar visualizaciones
   pca_figures['pca_2d'].show()
   pca_figures['pca_3d'].show()

analyze_clusters
^^^^^^^^^^^^^^^^

.. automethod:: src.clustering.ClusterAnalyzer.analyze_clusters
   :noindex:

Analiza perfiles de cada cluster.

**Parameters:**
   * ``df_with_clusters`` (DataFrame): DataFrame con columna 'cluster'
   * ``feature_names`` (List[str]): Características usadas

**Returns:**
   Diccionario con análisis por cluster

**Análisis por cluster:**
   * Tamaño y porcentaje del total
   * Estadísticas descriptivas de características
   * Distribución de nivel de riesgo

Funciones de Renderizado
-------------------------

render_clustering_analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.clustering.render_clustering_analysis

Renderiza el módulo completo de clustering en Streamlit.

**Funcionalidades:**
   * Selección de características
   * Configuración de algoritmos
   * Optimización de k
   * Ejecución de clustering
   * Visualizaciones PCA 2D/3D
   * Análisis de perfiles

render_clustering_module
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.clustering.render_clustering_module

Función principal para renderizar el módulo de clustering.

Métricas de Evaluación
-----------------------

Silhouette Score
^^^^^^^^^^^^^^^^

Mide qué tan similar es un objeto a su propio cluster comparado con otros clusters.

.. math::

   s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}

* **Rango:** [-1, 1]
* **Interpretación:**
   * Cerca de 1: Bien asignado
   * Cerca de 0: En el borde entre clusters
   * Negativo: Probablemente mal asignado

Davies-Bouldin Index
^^^^^^^^^^^^^^^^^^^^

Mide la similitud promedio entre cada cluster y su más similar.

* **Rango:** [0, ∞)
* **Interpretación:** Valores más bajos indican mejor clustering

Calinski-Harabasz Index
^^^^^^^^^^^^^^^^^^^^^^^^

Ratio de dispersión entre clusters vs dentro de clusters.

* **Rango:** [0, ∞)
* **Interpretación:** Valores más altos indican mejor clustering

Ejemplo Completo
----------------

.. code-block:: python

   from src.clustering import ClusterAnalyzer
   import pandas as pd
   
   # Cargar datos
   df = pd.read_csv("datos_con_caracteristicas.csv")
   
   # Crear analizador
   analyzer = ClusterAnalyzer(df)
   
   # Seleccionar características
   features = [
       'edad', 'salario_mensual', 'puntaje_datacredito',
       'dti', 'ltv', 'capacidad_residual'
   ]
   
   # Preparar datos
   X_scaled, valid_features = analyzer.prepare_data(features)
   print(f"Datos preparados: {X_scaled.shape}")
   
   # Encontrar k óptimo
   k_results = analyzer.find_optimal_k(X_scaled, max_k=10)
   optimal_k = k_results['optimal_k_methods']['silhouette']
   print(f"K óptimo: {optimal_k}")
   
   # Ejecutar K-Means
   results = analyzer.perform_clustering(X_scaled, 'kmeans', n_clusters=optimal_k)
   
   print(f"\nResultados del clustering:")
   print(f"  Silhouette Score: {results['metrics']['silhouette_score']:.3f}")
   print(f"  Davies-Bouldin: {results['metrics']['davies_bouldin_score']:.3f}")
   print(f"  Calinski-Harabasz: {results['metrics']['calinski_harabasz_score']:.0f}")
   
   # Agregar labels al DataFrame
   df['cluster'] = results['labels']
   
   # Analizar perfiles
   cluster_analysis = analyzer.analyze_clusters(df, valid_features)
   
   for cluster_id, analysis in cluster_analysis.items():
       print(f"\nCluster {cluster_id}:")
       print(f"  Tamaño: {analysis['size']} ({analysis['percentage']:.1f}%)")
       print(f"  Distribución de riesgo: {analysis['risk_distribution']}")
   
   # Crear visualizaciones PCA
   pca_figures = analyzer.create_pca_visualizations(
       X_scaled,
       results['labels'],
       valid_features
   )
   
   # Guardar resultados
   df.to_csv("datos_con_clusters.csv", index=False)

Ver también
-----------

* :doc:`feature_engineering` - Ingeniería de características
* :doc:`supervised_models` - Modelos supervisados
* :doc:`rbm_model` - RBM para clustering