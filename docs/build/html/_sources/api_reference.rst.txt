Referencia de la API
====================

Esta sección contiene la documentación completa de todas las clases, funciones y módulos del sistema.

.. toctree::
   :maxdepth: 3
   :caption: Módulos del Sistema:

   api/app
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

Índice de Módulos
-----------------

Módulo Principal
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/generated
   
   app

Módulos de Datos
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/generated
   
   generar_datos
   data_processor

Módulos de Análisis
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/generated
   
   univariate_analysis
   bivariate_analysis
   feature_engineering
   clustering

Módulos de Modelado
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/generated
   
   rbm_model
   supervised_models
   prediction
   retraining

Módulo Educativo
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/generated
   
   educational_rag

Índice de Clases
----------------

.. autosummary::
   :toctree: api/generated
   :recursive:
   
   generar_datos.GeneradorCreditoHipotecarioRealista
   data_processor.DataProcessor
   feature_engineering.FeatureEngineer
   clustering.ClusterAnalyzer
   rbm_model.RestrictedBoltzmannMachine
   supervised_models.SupervisedModelTrainer
   prediction.CreditRiskPredictor
   educational_rag.EducationalRAG
   retraining.ModelRetrainer

Índice de Funciones
-------------------

Funciones de Generación de Datos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: generar_datos.generar_datos_credito_realista

Funciones de Renderizado
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: data_processor.render_data_processor_module
.. autofunction:: univariate_analysis.render_univariate_module
.. autofunction:: bivariate_analysis.render_bivariate_module
.. autofunction:: feature_engineering.render_feature_engineering_module
.. autofunction:: clustering.render_clustering_module
.. autofunction:: rbm_model.render_rbm_module
.. autofunction:: supervised_models.render_supervised_models_module
.. autofunction:: prediction.render_prediction_module
.. autofunction:: educational_rag.render_educational_rag_module
.. autofunction:: retraining.render_retraining_module

Funciones de Visualización
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rbm_model.create_rbm_visualizations