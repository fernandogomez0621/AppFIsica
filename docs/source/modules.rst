Módulos del Sistema
==================

El sistema está organizado en módulos especializados que trabajan en conjunto
para proporcionar una solución completa de análisis de riesgo crediticio.

Módulos Principales
------------------

.. toctree::
   :maxdepth: 2

   modules/data_generation
   modules/data_processing
   modules/analysis
   modules/feature_engineering
   modules/clustering
   modules/rbm
   modules/supervised_models
   modules/prediction
   modules/educational_rag
   modules/retraining

Referencia Completa de la API
------------------------------

Documentación detallada de todos los módulos del código fuente:

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/generar_datos
   api/data_processor
   api/univariate_analysis
   api/bivariate_analysis
   api/feature_engineering
   api/clustering
   api/rbm_model
   api/supervised_models
   api/prediction
   api/educational_rag
   api/retraining

Arquitectura del Sistema
-----------------------

.. code-block:: text

   AppFIsica/
   │
   ├── app.py                          # 🚀 Aplicación principal
   ├── src/                           # 💻 Módulos de código
   │   ├── generar_datos.py           # 📊 Generación de datos
   │   ├── data_processor.py          # 🔧 Procesamiento
   │   ├── univariate_analysis.py     # 📈 Análisis univariado
   │   ├── bivariate_analysis.py      # 🔗 Análisis bivariado
   │   ├── feature_engineering.py     # ⚙️ Ingeniería características
   │   ├── clustering.py              # 🎯 Clustering
   │   ├── rbm_model.py               # ⚡ Máquina de Boltzmann
   │   ├── supervised_models.py       # 🤖 Modelos supervisados
   │   ├── prediction.py              # 🔮 Predicción
   │   └── educational_rag.py         # 🎓 Sistema RAG
   │
   ├── data/                          # 💾 Almacenamiento de datos
   ├── models/                        # 🧠 Modelos entrenados
   ├── articles/                      # 📚 Papers científicos
   └── docs/                          # 📖 Documentación

Flujo de Datos
--------------

.. mermaid::

   graph TD
       A[Datos Originales] --> B[Validación y Limpieza]
       B --> C[Análisis Exploratorio]
       C --> D[Ingeniería de Características]
       D --> E[Clustering]
       D --> F[Entrenamiento RBM]
       F --> G[Características RBM]
       D --> H[Modelos Supervisados]
       G --> H
       H --> I[Predicción]
       
       J[Papers PDF] --> K[Sistema RAG]
       K --> L[Chat Educativo]

Descripción de Módulos
---------------------

📊 Generación de Datos
~~~~~~~~~~~~~~~~~~~~~

**Propósito**: Crear datasets sintéticos realistas para Colombia.

**Características**:
- Correlaciones lógicas entre variables
- Distribuciones realistas (60% Bajo, 25% Medio, 15% Alto)
- Configuración de parámetros
- Exportación múltiple (CSV, Excel, Parquet)

**Uso**:

.. code-block:: python

   from src.generar_datos import generar_datos_credito_realista
   
   df = generar_datos_credito_realista(
       n_registros=10000,
       semilla=42
   )

🔧 Procesamiento de Datos
~~~~~~~~~~~~~~~~~~~~~~~~

**Propósito**: Validar y limpiar datos cargados por el usuario.

**Funcionalidades**:
- Carga de múltiples formatos
- Validaciones automáticas
- Detección de outliers
- Imputación inteligente
- Reportes de calidad

📈 Análisis Descriptivo
~~~~~~~~~~~~~~~~~~~~~~

**Propósito**: Análisis estadístico univariado y bivariado.

**Análisis Univariado**:
- Estadísticas completas (media, mediana, percentiles)
- Visualizaciones (histogramas, boxplots, Q-Q plots)
- Tests de normalidad
- Detección de outliers

**Análisis Bivariado**:
- Matrices de correlación
- Gráficos de dispersión
- Tablas de contingencia
- Tests estadísticos (Chi², ANOVA)

⚙️ Ingeniería de Características
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Propósito**: Crear variables derivadas automáticamente.

**Tipos de Características**:
- **Ratios Financieros**: LTV, DTI, capacidad ahorro
- **Indicadores de Riesgo**: Scores de edad, estabilidad, legal
- **Interacciones**: Educación×Salario, Edad×Empleo
- **Discretización**: Grupos de edad, rangos salariales
- **Transformaciones**: Log, raíz cuadrada

🎯 Clustering
~~~~~~~~~~~~

**Propósito**: Segmentación de solicitantes en grupos homogéneos.

**Algoritmos**:
- K-Means (rápido, esférico)
- Jerárquico (dendrograma)
- DBSCAN (detecta outliers)
- Gaussian Mixture (probabilístico)

**Visualizaciones**:
- PCA 2D/3D interactivo
- Determinación de K óptimo
- Perfiles por cluster

⚡ Máquina de Boltzmann Restringida
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Propósito**: Extracción de características latentes.

**Implementación**:
- RBM desde cero en NumPy
- Contrastive Divergence (CD-k)
- Métricas de evaluación
- Visualizaciones de pesos

**Ecuaciones Clave**:

.. math::

   P(h_j = 1 | v) = \sigma(b_j + \sum_i v_i W_{ij})

.. math::

   P(v_i = 1 | h) = \sigma(a_i + \sum_j h_j W_{ij})

🤖 Modelos Supervisados
~~~~~~~~~~~~~~~~~~~~~~

**Propósito**: Clasificación de riesgo crediticio.

**Modelos Incluidos**:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine
- Multi-Layer Perceptron

**Evaluación**:
- Validación cruzada 5-fold
- Múltiples métricas (Accuracy, F1, ROC-AUC)
- Matrices de confusión
- Importancia de características

🔮 Predicción
~~~~~~~~~~~~

**Propósito**: Evaluar riesgo de nuevos solicitantes.

**Características**:
- Formulario interactivo
- Validaciones en tiempo real
- Explicaciones detalladas
- Recomendaciones automáticas
- Historial de predicciones

🎓 Sistema RAG Educativo
~~~~~~~~~~~~~~~~~~~~~~~

**Propósito**: Aprendizaje interactivo sobre RBMs.

**Componentes**:
- Procesamiento de PDFs (PyMuPDF)
- Base vectorial (ChromaDB)
- Embeddings locales (HuggingFace)
- LLM (Groq Llama 3.3 70B)
- Chat interactivo

Configuración Avanzada
---------------------

Hiperparámetros RBM
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   rbm_config = {
       'n_hidden': 100,        # Unidades ocultas
       'learning_rate': 0.01,  # Tasa de aprendizaje
       'n_epochs': 100,        # Épocas
       'batch_size': 64,       # Tamaño de batch
       'k_cd': 1              # Pasos CD
   }

Configuración RAG
~~~~~~~~~~~~~~~~

.. code-block:: python

   rag_config = {
       'chunk_size': 1500,     # Tamaño de chunks
       'chunk_overlap': 300,   # Solapamiento
       'top_k_results': 6,     # Documentos relevantes
       'temperature': 0.3      # Creatividad LLM
   }

Optimización de Rendimiento
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Para Datasets Grandes (>50K registros)**:
- Usar muestreo estratificado
- Reducir épocas de RBM
- Usar validación holdout en lugar de CV

**Para Recursos Limitados**:
- Reducir ``n_hidden`` en RBM
- Usar menos modelos supervisados
- Desactivar visualizaciones 3D

Integración con Otros Sistemas
-----------------------------

API REST
~~~~~~~~

El sistema puede extenderse con una API REST:

.. code-block:: python

   from fastapi import FastAPI
   from src.prediction import CreditRiskPredictor
   
   app = FastAPI()
   predictor = CreditRiskPredictor()
   
   @app.post("/predict")
   def predict_risk(applicant_data: dict):
       return predictor.predict_risk(applicant_data)

Base de Datos
~~~~~~~~~~~~

Para producción, considera integrar con:
- PostgreSQL para datos transaccionales
- MongoDB para documentos no estructurados
- Redis para caché de predicciones

Monitoreo
~~~~~~~~

Implementa monitoreo de:
- Drift en distribuciones de datos
- Degradación de métricas de modelos
- Uso de recursos del sistema
- Latencia de predicciones