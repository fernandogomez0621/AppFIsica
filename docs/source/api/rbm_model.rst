rbm_model
=========

.. automodule:: src.rbm_model
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Implementación completa de Restricted Boltzmann Machine (RBM) para extracción de características latentes en datos de riesgo crediticio. Incluye entrenamiento con Contrastive Divergence, métricas de evaluación y visualizaciones de diagnóstico.

Clases Principales
------------------

RestrictedBoltzmannMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: src.rbm_model.RestrictedBoltzmannMachine
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Implementación completa de Máquina de Boltzmann Restringida (RBM) con algoritmo Contrastive Divergence.
   
   La RBM es un modelo generativo no supervisado que aprende representaciones latentes mediante una arquitectura de dos capas:
   
   * **Capa visible:** Representa los datos de entrada
   * **Capa oculta:** Captura características latentes
   
   **Parámetros del constructor:**
   
   :param n_visible: Número de unidades visibles (dimensión de entrada)
   :type n_visible: int
   :param n_hidden: Número de unidades ocultas (default: 100)
   :type n_hidden: int
   :param learning_rate: Tasa de aprendizaje (default: 0.01)
   :type learning_rate: float
   :param n_epochs: Número de épocas de entrenamiento (default: 100)
   :type n_epochs: int
   :param batch_size: Tamaño del batch (default: 64)
   :type batch_size: int
   :param k_cd: Pasos de Gibbs sampling en CD (default: 1)
   :type k_cd: int
   :param random_state: Semilla aleatoria (default: 42)
   :type random_state: int
   
   **Atributos:**
   
   * ``W``: Matriz de pesos (n_visible × n_hidden)
   * ``visible_bias``: Sesgos de capa visible
   * ``hidden_bias``: Sesgos de capa oculta
   * ``training_history``: Historia de entrenamiento
   * ``is_trained``: Estado del modelo
   * ``scaler``: Normalizador de datos
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.rbm_model import RestrictedBoltzmannMachine
      import numpy as np
      
      # Preparar datos
      X = df[numeric_features].values  # Shape: (n_samples, n_features)
      
      # Crear RBM
      rbm = RestrictedBoltzmannMachine(
          n_visible=X.shape[1],
          n_hidden=100,
          learning_rate=0.01,
          n_epochs=100,
          batch_size=64,
          k_cd=1
      )
      
      # Entrenar
      history = rbm.fit(X, validation_split=0.2)
      
      # Extraer características
      hidden_features = rbm.transform(X)
      
      # Reconstruir datos
      X_reconstructed = rbm.reconstruct(X)
      
      # Generar muestras sintéticas
      synthetic_samples = rbm.generate_samples(n_samples=100)

Métodos de Entrenamiento
-------------------------

fit
^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine.fit
   :noindex:

Entrena la RBM usando Contrastive Divergence.

**Parameters:**
   * ``X`` (np.ndarray): Datos de entrenamiento
   * ``validation_split`` (float): Proporción para validación (default: 0.2)
   * ``verbose`` (bool): Si mostrar progreso (default: True)

**Returns:**
   Historia de entrenamiento con métricas

**Proceso de entrenamiento:**

1. Normalizar datos con MinMaxScaler
2. Dividir en train/validation
3. Para cada época:
   
   * Shuffle de datos
   * Procesar batches con Contrastive Divergence
   * Actualizar pesos y sesgos
   * Calcular error de reconstrucción
   
4. Calcular métricas adicionales cada 10 épocas

**Métricas monitoreadas:**
   * Error de reconstrucción (MSE)
   * Pseudo log-likelihood
   * Energía libre

transform
^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine.transform
   :noindex:

Extrae características de la capa oculta.

**Parameters:**
   * ``X`` (np.ndarray): Datos de entrada

**Returns:**
   Activaciones de la capa oculta (n_samples × n_hidden)

**Uso:**

.. code-block:: python

   # Extraer características latentes
   hidden_features = rbm.transform(X)
   
   print(f"Características originales: {X.shape[1]}")
   print(f"Características latentes: {hidden_features.shape[1]}")
   
   # Usar en modelos supervisados
   from sklearn.ensemble import RandomForestClassifier
   
   clf = RandomForestClassifier()
   clf.fit(hidden_features, y)

reconstruct
^^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine.reconstruct
   :noindex:

Reconstruye datos desde la representación oculta.

**Parameters:**
   * ``X`` (np.ndarray): Datos de entrada

**Returns:**
   Datos reconstruidos

**Uso para evaluación:**

.. code-block:: python

   # Reconstruir datos
   X_reconstructed = rbm.reconstruct(X_test)
   
   # Calcular error de reconstrucción
   from sklearn.metrics import mean_squared_error
   
   mse = mean_squared_error(X_test, X_reconstructed)
   print(f"Error de reconstrucción: {mse:.6f}")

generate_samples
^^^^^^^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine.generate_samples
   :noindex:

Genera muestras sintéticas usando Gibbs sampling.

**Parameters:**
   * ``n_samples`` (int): Número de muestras a generar (default: 100)
   * ``n_gibbs`` (int): Pasos de Gibbs sampling (default: 1000)

**Returns:**
   Muestras generadas (n_samples × n_visible)

**Ejemplo:**

.. code-block:: python

   # Generar 500 muestras sintéticas
   synthetic_data = rbm.generate_samples(n_samples=500, n_gibbs=1000)
   
   # Crear DataFrame
   df_synthetic = pd.DataFrame(synthetic_data, columns=feature_names)
   
   # Comparar distribuciones
   print("Estadísticas - Datos reales:")
   print(df[feature_names].describe())
   
   print("\nEstadísticas - Datos sintéticos:")
   print(df_synthetic.describe())

Métodos de Persistencia
------------------------

save_model
^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine.save_model
   :noindex:

Guarda el modelo entrenado en disco.

**Parameters:**
   * ``filepath`` (str): Ruta del archivo
   * ``feature_names`` (List[str], optional): Nombres de características

**Ejemplo:**

.. code-block:: python

   rbm.save_model(
       filepath="models/rbm/rbm_h100_lr0.01_e100.pkl",
       feature_names=feature_names
   )

load_model
^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine.load_model
   :noindex:

Carga un modelo entrenado desde disco.

**Parameters:**
   * ``filepath`` (str): Ruta del archivo

**Returns:**
   Instancia de RBM cargada

**Ejemplo:**

.. code-block:: python

   # Cargar modelo
   rbm_loaded = RestrictedBoltzmannMachine.load_model(
       "models/rbm/rbm_h100_lr0.01_e100.pkl"
   )
   
   # Usar modelo cargado
   hidden_features = rbm_loaded.transform(X_new)

Funciones de Visualización
---------------------------

create_rbm_visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.rbm_model.create_rbm_visualizations

Crea visualizaciones de diagnóstico para la RBM.

**Parameters:**
   * ``rbm`` (RestrictedBoltzmannMachine): Modelo entrenado
   * ``X_original`` (np.ndarray): Datos originales
   * ``feature_names`` (List[str], optional): Nombres de características

**Returns:**
   Diccionario con figuras de Plotly

**Visualizaciones generadas:**

1. **learning_curve:** Curva de aprendizaje (error vs época)
2. **weights_heatmap:** Heatmap de matriz de pesos
3. **activations_dist:** Distribución de activaciones ocultas
4. **reconstruction_comparison:** Original vs Reconstruido

Funciones de Renderizado
-------------------------

render_rbm_module
^^^^^^^^^^^^^^^^^

.. autofunction:: src.rbm_model.render_rbm_module

Renderiza el módulo completo de RBM en Streamlit.

**Funcionalidades:**
   * Configuración de arquitectura
   * Entrenamiento interactivo
   * Visualizaciones de diagnóstico
   * Extracción de características
   * Generación de muestras sintéticas

Fundamentos Teóricos
--------------------

Función de Energía
^^^^^^^^^^^^^^^^^^

La RBM define una función de energía:

.. math::

   E(v,h) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i W_{ij} h_j

Donde:
   * :math:`v`: Vector de unidades visibles
   * :math:`h`: Vector de unidades ocultas
   * :math:`a`: Sesgos de capa visible
   * :math:`b`: Sesgos de capa oculta
   * :math:`W`: Matriz de pesos

Distribución de Probabilidad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   P(v,h) = \frac{1}{Z} e^{-E(v,h)}

Donde :math:`Z` es la función de partición (intratable).

Probabilidades Condicionales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   P(h_j=1|v) = \sigma(b_j + \sum_i v_i W_{ij})

.. math::

   P(v_i=1|h) = \sigma(a_i + \sum_j h_j W_{ij})

Donde :math:`\sigma(x) = \frac{1}{1 + e^{-x}}` es la función sigmoide.

Contrastive Divergence
^^^^^^^^^^^^^^^^^^^^^^

Algoritmo de entrenamiento CD-k:

1. **Fase positiva:** Calcular :math:`P(h|v^{(0)})`
2. **Fase negativa:** 
   
   * Muestrear :math:`h^{(0)} \sim P(h|v^{(0)})`
   * Para k pasos: :math:`v^{(k)} \sim P(v|h^{(k-1)})`, :math:`h^{(k)} \sim P(h|v^{(k)})`
   
3. **Actualización:**

.. math::

   \Delta W_{ij} = \eta (v_i^{(0)} h_j^{(0)} - v_i^{(k)} h_j^{(k)})

Ejemplo Completo
----------------

.. code-block:: python

   from src.rbm_model import RestrictedBoltzmannMachine, create_rbm_visualizations
   import pandas as pd
   import numpy as np
   
   # Cargar y preparar datos
   df = pd.read_csv("datos_credito_hipotecario_realista.csv")
   
   # Seleccionar características numéricas
   numeric_features = [
       'edad', 'salario_mensual', 'puntaje_datacredito',
       'dti', 'ltv', 'capacidad_residual', 'patrimonio_total'
   ]
   
   X = df[numeric_features].values
   
   # Crear y entrenar RBM
   rbm = RestrictedBoltzmannMachine(
       n_visible=len(numeric_features),
       n_hidden=50,
       learning_rate=0.01,
       n_epochs=100,
       batch_size=64,
       k_cd=1,
       random_state=42
   )
   
   # Entrenar
   history = rbm.fit(X, validation_split=0.2, verbose=True)
   
   print(f"Error final: {history['reconstruction_error'][-1]:.6f}")
   
   # Extraer características latentes
   hidden_features = rbm.transform(X)
   print(f"Características latentes: {hidden_features.shape}")
   
   # Crear DataFrame enriquecido
   feature_names_rbm = [f"RBM_H{i+1}" for i in range(hidden_features.shape[1])]
   df_rbm = pd.DataFrame(hidden_features, columns=feature_names_rbm)
   df_enhanced = pd.concat([df.reset_index(drop=True), df_rbm], axis=1)
   
   # Generar muestras sintéticas
   synthetic_samples = rbm.generate_samples(n_samples=100, n_gibbs=1000)
   
   # Crear visualizaciones
   figures = create_rbm_visualizations(rbm, X, numeric_features)
   figures['learning_curve'].show()
   figures['weights_heatmap'].show()
   
   # Guardar modelo
   rbm.save_model("models/rbm/rbm_model.pkl", feature_names=numeric_features)
   
   # Cargar modelo
   rbm_loaded = RestrictedBoltzmannMachine.load_model("models/rbm/rbm_model.pkl")

Métodos Internos
----------------

_initialize_parameters
^^^^^^^^^^^^^^^^^^^^^^

Inicializa pesos y sesgos de la RBM usando inicialización Xavier/Glorot.

.. math::

   W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{visible} + n_{hidden}}})

_sigmoid
^^^^^^^^

Función sigmoide estable numéricamente.

.. math::

   \sigma(x) = \begin{cases}
   \frac{1}{1 + e^{-x}} & \text{si } x \geq 0 \\
   \frac{e^x}{1 + e^x} & \text{si } x < 0
   \end{cases}

_sample_hidden
^^^^^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine._sample_hidden
   :noindex:

Muestrea unidades ocultas dado el estado visible.

**Parameters:**
   * ``visible`` (np.ndarray): Estado de unidades visibles

**Returns:**
   Tupla (probabilidades_ocultas, estados_ocultos)

_sample_visible
^^^^^^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine._sample_visible
   :noindex:

Muestrea unidades visibles dado el estado oculto.

**Parameters:**
   * ``hidden`` (np.ndarray): Estado de unidades ocultas

**Returns:**
   Tupla (probabilidades_visibles, estados_visibles)

_contrastive_divergence
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine._contrastive_divergence
   :noindex:

Implementa el algoritmo Contrastive Divergence (CD-k).

**Parameters:**
   * ``batch`` (np.ndarray): Batch de datos de entrenamiento

**Returns:**
   Diccionario con gradientes para actualizar parámetros

**Algoritmo CD-k:**

1. **Fase positiva:** :math:`h^{(0)} \sim P(h|v^{data})`
2. **Fase negativa:** k pasos de Gibbs sampling
3. **Gradientes:**

.. math::

   \Delta W = \eta \frac{1}{N} (v^{data} h^{(0)T} - v^{(k)} h^{(k)T})

_compute_reconstruction_error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calcula el error de reconstrucción (MSE).

_compute_pseudo_log_likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine._compute_pseudo_log_likelihood
   :noindex:

Calcula pseudo log-likelihood como aproximación de la verosimilitud.

**Parameters:**
   * ``data`` (np.ndarray): Datos de evaluación
   * ``n_samples`` (int): Número de muestras para estimación (default: 100)

**Returns:**
   Pseudo log-likelihood promedio

_free_energy
^^^^^^^^^^^^

.. automethod:: src.rbm_model.RestrictedBoltzmannMachine._free_energy
   :noindex:

Calcula la energía libre.

.. math::

   F(v) = -\log \sum_h e^{-E(v,h)} = -a^T v - \sum_j \log(1 + e^{b_j + W_j^T v})

Funciones de Visualización
---------------------------

create_rbm_visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.rbm_model.create_rbm_visualizations

Crea visualizaciones completas de diagnóstico.

**Visualizaciones:**

1. **Curva de aprendizaje:** Error vs época
2. **Heatmap de pesos:** Visualización de matriz W
3. **Distribución de activaciones:** Histograma de activaciones ocultas
4. **Comparación reconstrucción:** Original vs reconstruido

Funciones de Renderizado
-------------------------

render_rbm_module
^^^^^^^^^^^^^^^^^

.. autofunction:: src.rbm_model.render_rbm_module

Renderiza el módulo completo de RBM en Streamlit.

**Funcionalidades:**
   * Configuración interactiva de hiperparámetros
   * Entrenamiento con progreso en tiempo real
   * Visualizaciones de diagnóstico
   * Extracción de características
   * Generación de datos sintéticos
   * Guardado y carga de modelos

Hiperparámetros Recomendados
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parámetro
     - Valor Recomendado
     - Descripción
   * - n_hidden
     - 50-200
     - Más unidades capturan más patrones
   * - learning_rate
     - 0.001-0.01
     - Tasas altas pueden causar inestabilidad
   * - n_epochs
     - 100-200
     - Monitorear convergencia
   * - batch_size
     - 32-128
     - Balance entre velocidad y estabilidad
   * - k_cd
     - 1-3
     - CD-1 suele ser suficiente

Interpretación de Métricas
---------------------------

Error de Reconstrucción
^^^^^^^^^^^^^^^^^^^^^^^

* **< 0.01:** Excelente reconstrucción
* **0.01-0.05:** Buena reconstrucción
* **> 0.05:** Revisar hiperparámetros

Sparsity de Activaciones
^^^^^^^^^^^^^^^^^^^^^^^^^

Porcentaje de activaciones < 0.1:

* **< 20%:** Baja sparsity (todas las unidades activas)
* **20-50%:** Sparsity moderada (ideal)
* **> 50%:** Alta sparsity (muchas unidades inactivas)

Ver también
-----------

* :doc:`feature_engineering` - Ingeniería de características
* :doc:`supervised_models` - Modelos supervisados
* :doc:`clustering` - Clustering con RBM
* :doc:`educational_rag` - Aprende sobre RBMs