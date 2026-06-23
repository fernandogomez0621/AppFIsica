supervised_models
=================

.. automodule:: src.supervised_models
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Módulo de entrenamiento y evaluación de múltiples modelos de clasificación de riesgo crediticio con integración de características RBM y optimización de hiperparámetros.

Clases Principales
------------------

SupervisedModelTrainer
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: src.supervised_models.SupervisedModelTrainer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Entrenador de modelos supervisados con soporte para múltiples algoritmos y optimización automática.
   
   **Atributos:**
   
   * ``models``: Diccionario de modelos entrenados
   * ``results``: Resultados de evaluación
   * ``X_train, X_test, X_holdout``: Conjuntos de datos
   * ``y_train, y_test, y_holdout``: Variables objetivo
   * ``feature_names``: Nombres de características
   * ``label_encoder``: Codificador de etiquetas
   * ``scaler``: Escalador de características
   * ``model_configs``: Configuraciones de modelos
   
   **Modelos soportados:**
   
   1. **Logistic Regression:** Modelo lineal probabilístico
   2. **Random Forest:** Ensemble de árboles de decisión
   3. **XGBoost:** Gradient boosting optimizado
   4. **LightGBM:** Gradient boosting eficiente
   5. **SVM:** Support Vector Machine
   6. **MLP:** Multi-Layer Perceptron (red neuronal)
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.supervised_models import SupervisedModelTrainer
      import pandas as pd
      
      # Cargar datos
      df = pd.read_csv("datos_con_caracteristicas.csv")
      
      # Crear entrenador
      trainer = SupervisedModelTrainer()
      
      # Preparar datos
      trainer.prepare_data(df, target_col='nivel_riesgo')
      
      # Entrenar modelo
      results = trainer.train_model('xgboost', use_grid_search=True)
      
      print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
      print(f"F1-Score: {results['metrics']['f1_weighted']:.4f}")

Métodos de Preparación
----------------------

prepare_data
^^^^^^^^^^^^

.. automethod:: src.supervised_models.SupervisedModelTrainer.prepare_data
   :noindex:

Prepara datos para entrenamiento con división estratificada.

**Parameters:**
   * ``df`` (DataFrame): DataFrame con datos
   * ``target_col`` (str): Variable objetivo (default: 'nivel_riesgo')
   * ``test_size`` (float): Proporción para testing (default: 0.2)
   * ``holdout_size`` (float): Proporción para holdout (default: 0.1)

**Returns:**
   True si exitoso

**División de datos:**
   * 70% Entrenamiento (con validación cruzada 5-fold)
   * 20% Testing (evaluación final)
   * 10% Holdout (simulación de producción)

**Ejemplo:**

.. code-block:: python

   success = trainer.prepare_data(
       df,
       target_col='nivel_riesgo',
       test_size=0.2,
       holdout_size=0.1
   )
   
   if success:
       print(f"Train: {trainer.X_train.shape}")
       print(f"Test: {trainer.X_test.shape}")
       print(f"Holdout: {trainer.X_holdout.shape}")
       print(f"Características: {len(trainer.feature_names)}")

Métodos de Entrenamiento
-------------------------

train_model
^^^^^^^^^^^

.. automethod:: src.supervised_models.SupervisedModelTrainer.train_model
   :noindex:

Entrena un modelo específico con optimización opcional.

**Parameters:**
   * ``model_key`` (str): Clave del modelo ('logistic', 'random_forest', 'xgboost', etc.)
   * ``use_grid_search`` (bool): Si usar búsqueda de hiperparámetros (default: True)

**Returns:**
   Diccionario con resultados del entrenamiento

**Proceso:**

1. Cargar configuración del modelo
2. GridSearchCV si use_grid_search=True
3. Entrenar con mejores parámetros
4. Calcular métricas completas
5. Guardar modelo y métricas

**Ejemplo:**

.. code-block:: python

   # Entrenar con optimización
   results = trainer.train_model('xgboost', use_grid_search=True)
   
   print(f"Mejores parámetros: {results['best_params']}")
   print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
   print(f"F1-Score: {results['metrics']['f1_weighted']:.4f}")
   print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")

train_all_models
^^^^^^^^^^^^^^^^

.. automethod:: src.supervised_models.SupervisedModelTrainer.train_all_models
   :noindex:

Entrena todos los modelos seleccionados.

**Parameters:**
   * ``selected_models`` (List[str]): Lista de modelos a entrenar
   * ``use_grid_search`` (bool): Si usar optimización (default: True)

**Returns:**
   Diccionario con resultados de todos los modelos

**Ejemplo:**

.. code-block:: python

   # Entrenar múltiples modelos
   models_to_train = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
   
   all_results = trainer.train_all_models(
       selected_models=models_to_train,
       use_grid_search=True
   )
   
   # Comparar resultados
   for model_key, results in all_results.items():
       print(f"{model_key}:")
       print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
       print(f"  F1-Score: {results['metrics']['f1_weighted']:.4f}")

Métodos de Evaluación
----------------------

_calculate_metrics
^^^^^^^^^^^^^^^^^^

Calcula métricas de evaluación completas.

**Métricas calculadas:**

* **Accuracy:** Precisión global
* **Precision:** Macro y weighted
* **Recall:** Macro y weighted
* **F1-Score:** Macro y weighted
* **Cohen's Kappa:** Acuerdo ajustado por azar
* **Matthews Correlation:** Correlación de Matthews
* **ROC-AUC:** Área bajo curva ROC
* **Confusion Matrix:** Matriz de confusión
* **Classification Report:** Reporte detallado por clase

**Ejemplo:**

.. code-block:: python

   from sklearn.metrics import classification_report
   
   # Las métricas se calculan automáticamente
   metrics = results['metrics']
   
   print(f"Accuracy: {metrics['accuracy']:.4f}")
   print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
   print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
   print(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
   print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
   print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
   
   # Matriz de confusión
   print("\nMatriz de Confusión:")
   print(metrics['confusion_matrix'])
   
   # Reporte por clase
   print("\nReporte de Clasificación:")
   for class_name, class_metrics in metrics['classification_report'].items():
       if isinstance(class_metrics, dict):
           print(f"{class_name}: {class_metrics}")

Métodos de Visualización
-------------------------

create_comparison_visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.supervised_models.SupervisedModelTrainer.create_comparison_visualizations
   :noindex:

Crea visualizaciones comparativas de múltiples modelos.

**Parameters:**
   * ``results`` (Dict): Resultados de múltiples modelos

**Returns:**
   Diccionario con figuras de Plotly

**Visualizaciones:**
   * **model_comparison:** Barras comparativas de métricas
   * **roc_curves:** Curvas ROC superpuestas

create_confusion_matrix_plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.supervised_models.SupervisedModelTrainer.create_confusion_matrix_plot
   :noindex:

Crea visualización de matriz de confusión.

**Parameters:**
   * ``model_key`` (str): Clave del modelo
   * ``results`` (Dict): Resultados del modelo

**Returns:**
   Figura de Plotly con heatmap

Funciones de Renderizado
-------------------------

render_supervised_models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.supervised_models.render_supervised_models

Renderiza el módulo completo de modelos supervisados en Streamlit.

render_supervised_models_module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.supervised_models.render_supervised_models_module

Función principal para renderizar el módulo.

Configuración de Modelos
-------------------------

Logistic Regression
^^^^^^^^^^^^^^^^^^^

**Hiperparámetros optimizados:**
   * C: [0.1, 1.0, 10.0]
   * penalty: ['l1', 'l2']
   * solver: ['liblinear']

Random Forest
^^^^^^^^^^^^^

**Hiperparámetros optimizados:**
   * n_estimators: [100, 200]
   * max_depth: [10, 20, None]
   * min_samples_split: [2, 5]

XGBoost
^^^^^^^

**Hiperparámetros optimizados:**
   * n_estimators: [100, 200]
   * max_depth: [3, 6]
   * learning_rate: [0.01, 0.1]

LightGBM
^^^^^^^^

**Hiperparámetros optimizados:**
   * n_estimators: [100, 200]
   * max_depth: [3, 6]
   * learning_rate: [0.01, 0.1]

SVM
^^^

**Hiperparámetros optimizados:**
   * C: [0.1, 1.0, 10.0]
   * kernel: ['rbf', 'linear']

MLP
^^^

**Hiperparámetros optimizados:**
   * hidden_layer_sizes: [(100,), (100, 50)]
   * alpha: [0.001, 0.01]
   * learning_rate: ['constant', 'adaptive']

Ejemplo Completo
----------------

.. code-block:: python

   from src.supervised_models import SupervisedModelTrainer
   import pandas as pd
   
   # Cargar datos con características RBM
   df = pd.read_csv("datos_con_rbm.csv")
   
   # Crear entrenador
   trainer = SupervisedModelTrainer()
   
   # Preparar datos
   trainer.prepare_data(df, target_col='nivel_riesgo')
   
   # Entrenar múltiples modelos
   models = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
   all_results = trainer.train_all_models(models, use_grid_search=True)
   
   # Comparar modelos
   print("Comparación de Modelos:")
   print("-" * 60)
   
   for model_key, results in all_results.items():
       metrics = results['metrics']
       print(f"\n{trainer.model_configs[model_key]['name']}:")
       print(f"  Accuracy:  {metrics['accuracy']:.4f}")
       print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
       print(f"  Precision: {metrics['precision_weighted']:.4f}")
       print(f"  Recall:    {metrics['recall_weighted']:.4f}")
       print(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
       print(f"  Kappa:     {metrics['cohen_kappa']:.4f}")
   
   # Crear visualizaciones comparativas
   comparison_figs = trainer.create_comparison_visualizations(all_results)
   comparison_figs['model_comparison'].show()
   comparison_figs['roc_curves'].show()
   
   # Analizar mejor modelo
   best_model_key = max(
       all_results.keys(),
       key=lambda k: all_results[k]['metrics']['f1_weighted']
   )
   
   print(f"\n🏆 Mejor modelo: {trainer.model_configs[best_model_key]['name']}")
   
   best_results = all_results[best_model_key]
   
   # Matriz de confusión
   fig_cm = trainer.create_confusion_matrix_plot(best_model_key, best_results)
   fig_cm.show()
   
   # Importancia de características (si disponible)
   if hasattr(best_results['model'], 'feature_importances_'):
       importances = best_results['model'].feature_importances_
       feature_importance_df = pd.DataFrame({
           'Feature': trainer.feature_names,
           'Importance': importances
       }).sort_values('Importance', ascending=False)
       
       print("\nTop 10 Características:")
       print(feature_importance_df.head(10))

Interpretación de Métricas
---------------------------

Accuracy
^^^^^^^^

Proporción de predicciones correctas.

.. math::

   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

* **> 0.90:** Excelente
* **0.80-0.90:** Muy bueno
* **0.70-0.80:** Bueno
* **< 0.70:** Revisar modelo

Precision
^^^^^^^^^

Proporción de predicciones positivas correctas.

.. math::

   \text{Precision} = \frac{TP}{TP + FP}

Recall (Sensibilidad)
^^^^^^^^^^^^^^^^^^^^^

Proporción de positivos reales detectados.

.. math::

   \text{Recall} = \frac{TP}{TP + FN}

F1-Score
^^^^^^^^

Media armónica de precision y recall.

.. math::

   F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}

Cohen's Kappa
^^^^^^^^^^^^^

Acuerdo entre predicciones y realidad, ajustado por azar.

* **< 0:** Peor que azar
* **0-0.20:** Acuerdo leve
* **0.21-0.40:** Acuerdo justo
* **0.41-0.60:** Acuerdo moderado
* **0.61-0.80:** Acuerdo sustancial
* **0.81-1.00:** Acuerdo casi perfecto

ROC-AUC
^^^^^^^

Área bajo la curva ROC (Receiver Operating Characteristic).

* **0.90-1.00:** Excelente
* **0.80-0.90:** Muy bueno
* **0.70-0.80:** Bueno
* **0.60-0.70:** Pobre
* **0.50-0.60:** Falla

Funciones de Renderizado
-------------------------

render_supervised_models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.supervised_models.render_supervised_models

Renderiza el módulo completo de modelos supervisados en Streamlit.

**Funcionalidades:**
   * Selección de dataset (original, con características, con RBM)
   * Selección de modelos a entrenar
   * Optimización de hiperparámetros
   * Entrenamiento con barra de progreso
   * Comparación de modelos
   * Análisis detallado por modelo
   * Visualización de importancia de características

render_supervised_models_module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.supervised_models.render_supervised_models_module

Función principal para renderizar el módulo.

Ejemplo de Optimización de Hiperparámetros
-------------------------------------------

.. code-block:: python

   from src.supervised_models import SupervisedModelTrainer
   from sklearn.model_selection import GridSearchCV
   
   # Crear entrenador
   trainer = SupervisedModelTrainer()
   trainer.prepare_data(df)
   
   # Configuración personalizada para XGBoost
   custom_params = {
       'n_estimators': [50, 100, 200, 300],
       'max_depth': [3, 4, 5, 6, 7],
       'learning_rate': [0.01, 0.05, 0.1, 0.2],
       'subsample': [0.8, 0.9, 1.0],
       'colsample_bytree': [0.8, 0.9, 1.0]
   }
   
   # Actualizar configuración
   trainer.model_configs['xgboost']['params'] = custom_params
   
   # Entrenar con búsqueda exhaustiva
   results = trainer.train_model('xgboost', use_grid_search=True)
   
   print(f"Mejores hiperparámetros encontrados:")
   for param, value in results['best_params'].items():
       print(f"  {param}: {value}")

Ejemplo de Validación Cruzada
------------------------------

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   
   # Entrenar modelo
   results = trainer.train_model('random_forest')
   model = results['model']
   
   # Validación cruzada 5-fold
   cv_scores = cross_val_score(
       model,
       trainer.X_train_scaled,
       trainer.y_train,
       cv=5,
       scoring='f1_weighted'
   )
   
   print(f"F1-Scores por fold: {cv_scores}")
   print(f"Media: {cv_scores.mean():.4f}")
   print(f"Desv. Estándar: {cv_scores.std():.4f}")

Ejemplo de Análisis de Errores
-------------------------------

.. code-block:: python

   import numpy as np
   
   # Obtener predicciones
   y_pred = results['predictions']['y_test_pred']
   y_true = trainer.y_test
   
   # Identificar errores
   errors_mask = y_pred != y_true
   errors_indices = np.where(errors_mask)[0]
   
   print(f"Total de errores: {errors_mask.sum()}")
   print(f"Tasa de error: {errors_mask.sum() / len(y_true):.2%}")
   
   # Analizar errores por clase
   for class_idx, class_name in enumerate(trainer.label_encoder.classes_):
       class_mask = y_true == class_idx
       class_errors = errors_mask[class_mask].sum()
       class_total = class_mask.sum()
       
       print(f"\nClase '{class_name}':")
       print(f"  Total: {class_total}")
       print(f"  Errores: {class_errors}")
       print(f"  Tasa de error: {class_errors/class_total:.2%}")
   
   # Ver casos mal clasificados
   X_test_df = pd.DataFrame(
       trainer.X_test,
       columns=trainer.feature_names
   )
   
   errors_df = X_test_df.iloc[errors_indices].copy()
   errors_df['true_class'] = trainer.label_encoder.inverse_transform(y_true[errors_indices])
   errors_df['predicted_class'] = trainer.label_encoder.inverse_transform(y_pred[errors_indices])
   
   print("\nPrimeros 5 casos mal clasificados:")
   print(errors_df[['true_class', 'predicted_class']].head())

Guardado de Modelos
-------------------

Los modelos se guardan automáticamente en:

* **Modelo:** ``models/supervised/{model_key}_model.pkl``
* **Métricas:** ``models/supervised/{model_key}_metrics.json``

**Contenido del archivo de modelo:**

.. code-block:: python

   {
       'model': trained_model,
       'scaler': StandardScaler,
       'label_encoder': LabelEncoder,
       'feature_names': list,
       'best_params': dict,
       'metrics': dict,
       'timestamp': str
   }

Ver también
-----------

* :doc:`feature_engineering` - Ingeniería de características
* :doc:`rbm_model` - Características RBM
* :doc:`prediction` - Sistema de predicción
* :doc:`retraining` - Reentrenamiento de modelos