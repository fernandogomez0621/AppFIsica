====================================
8. Modelos Supervisados
====================================

Esta guía te enseñará a entrenar y comparar múltiples modelos de clasificación supervisada para predecir el nivel de riesgo crediticio.

Objetivo del Módulo
===================

El módulo de modelos supervisados te permite:

* 🤖 **Entrenar 9 algoritmos** de clasificación
* 📊 **Comparar rendimiento** automáticamente
* 🎯 **Optimizar hiperparámetros** con Grid Search
* 📈 **Visualizar métricas** y curvas ROC
* 💾 **Guardar mejores modelos** para producción
* 🔍 **Analizar importancia** de features

Acceso al Módulo
================

En el sidebar, click en:

.. code-block:: text

   🤖 Modelado → 🤖 Modelos Supervisados

Algoritmos Disponibles
=======================

1. Logistic Regression
-----------------------

**Características:**

* Simple y rápido
* Interpretable
* Baseline sólido

**Cuándo usar:** Relaciones lineales, interpretabilidad crítica.

2. Random Forest
----------------

**Características:**

* Robusto a outliers
* Maneja no linealidades
* Feature importance

**Cuándo usar:** Datos complejos, alta precisión.

3. XGBoost
----------

**Características:**

* Estado del arte
* Muy preciso
* Maneja desbalanceo

**Cuándo usar:** Competencias, máxima precisión.

4. LightGBM
-----------

**Características:**

* Muy rápido
* Eficiente en memoria
* Preciso

**Cuándo usar:** Datasets grandes, velocidad crítica.

5. Support Vector Machine (SVM)
--------------------------------

**Características:**

* Kernel tricks
* Margen máximo
* Robusto

**Cuándo usar:** Datasets pequeños/medianos, alta dimensionalidad.

6. K-Nearest Neighbors (KNN)
-----------------------------

**Características:**

* No paramétrico
* Simple
* Lazy learning

**Cuándo usar:** Patrones locales, pocos datos.

7. Naive Bayes
--------------

**Características:**

* Muy rápido
* Probabilístico
* Asume independencia

**Cuándo usar:** Baseline rápido, features independientes.

8. Neural Network (MLP)
-----------------------

**Características:**

* Aprende no linealidades
* Flexible
* Requiere más datos

**Cuándo usar:** Patrones complejos, suficientes datos.

9. Gradient Boosting
--------------------

**Características:**

* Ensemble potente
* Preciso
* Interpretable

**Cuándo usar:** Balance precisión/interpretabilidad.

Proceso de Entrenamiento
=========================

Paso 1: Preparar Datos
-----------------------

.. code-block:: text

   [⚙️ Preparar Datos]
   
   ✓ Features: 20 (incluyendo RBM)
   ✓ Target: nivel_riesgo (3 clases)
   ✓ Train: 8,000 (80%)
   ✓ Test: 2,000 (20%)
   ✓ Balanceo: SMOTE aplicado

Paso 2: Seleccionar Modelos
----------------------------

.. code-block:: text

   Modelos a entrenar:
   
   ☑ Logistic Regression
   ☑ Random Forest
   ☑ XGBoost
   ☑ LightGBM
   ☑ SVM
   ☐ KNN
   ☐ Naive Bayes
   ☐ Neural Network
   ☐ Gradient Boosting

Paso 3: Entrenar Modelos
-------------------------

.. code-block:: text

   [🎯 Entrenar Modelos Seleccionados]
   
   Entrenando Logistic Regression... ✓ (2s)
   Entrenando Random Forest... ✓ (15s)
   Entrenando XGBoost... ✓ (25s)
   Entrenando LightGBM... ✓ (12s)
   Entrenando SVM... ✓ (45s)
   
   ✓ Todos los modelos entrenados
   Tiempo total: 1m 39s

Comparación de Modelos
======================

Tabla de Métricas
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15 15

   * - Modelo
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - ROC-AUC
   * - XGBoost
     - **95.2%**
     - 94.8%
     - 95.1%
     - **95.0%**
     - **0.982**
   * - LightGBM
     - 94.8%
     - **95.1%**
     - 94.5%
     - 94.8%
     - 0.979
   * - Random Forest
     - 94.1%
     - 93.8%
     - **95.3%**
     - 94.5%
     - 0.975
   * - SVM
     - 92.5%
     - 92.1%
     - 92.8%
     - 92.4%
     - 0.968
   * - Logistic Reg.
     - 89.3%
     - 88.7%
     - 89.9%
     - 89.3%
     - 0.945

**Mejor modelo:** XGBoost (95.2% accuracy)

Matriz de Confusión
-------------------

Para XGBoost:

.. code-block:: text

                Predicho
              Bajo  Medio  Alto
   Real Bajo  1180    25     5
        Medio   30   570    15
        Alto     8    12   155
   
   Accuracy: 95.2%
   Errores: 95 de 2,000

Curvas ROC
----------

.. code-block:: text

   [📊 Visualizar Curvas ROC]
   
   Curva ROC multiclase (One-vs-Rest)
   • Bajo vs Rest: AUC = 0.985
   • Medio vs Rest: AUC = 0.978
   • Alto vs Rest: AUC = 0.983

Importancia de Features
=======================

Top 10 Features
---------------

Para Random Forest:

.. code-block:: text

   Feature Importance:
   
   1. dti: 0.185 ████████████████████
   2. puntaje_datacredito: 0.152 ████████████████
   3. capacidad_residual: 0.128 █████████████
   4. ltv: 0.095 ██████████
   5. rbm_feature_1: 0.082 ████████
   6. score_estabilidad: 0.071 ███████
   7. patrimonio_total: 0.065 ██████
   8. salario_mensual: 0.058 ██████
   9. rbm_feature_5: 0.052 █████
   10. edad: 0.048 ████

**Insights:**

* DTI es el predictor más importante
* Features RBM aportan valor significativo
* Variables financieras dominan

SHAP Values
-----------

.. code-block:: text

   [🔍 Calcular SHAP Values]
   
   Análisis de contribución individual
   Muestra cómo cada feature afecta predicciones

Optimización de Hiperparámetros
================================

Grid Search
-----------

.. code-block:: text

   Modelo: XGBoost
   
   Parámetros a optimizar:
   • n_estimators: [100, 200, 300]
   • max_depth: [3, 5, 7]
   • learning_rate: [0.01, 0.05, 0.1]
   • subsample: [0.8, 0.9, 1.0]
   
   [🔧 Ejecutar Grid Search]
   
   Probando 36 combinaciones...
   Mejor combinación encontrada:
   • n_estimators: 200
   • max_depth: 5
   • learning_rate: 0.05
   • subsample: 0.9
   
   Accuracy mejorada: 95.2% → 95.8%

Random Search
-------------

Más rápido que Grid Search:

.. code-block:: text

   [🎲 Ejecutar Random Search]
   
   100 iteraciones aleatorias
   Tiempo: 15 minutos
   Mejora: +0.4% accuracy

Validación Cruzada
==================

K-Fold Cross-Validation
-----------------------

.. code-block:: text

   [✅ Validación Cruzada (5-fold)]
   
   Fold 1: 94.8%
   Fold 2: 95.1%
   Fold 3: 95.5%
   Fold 4: 94.9%
   Fold 5: 95.3%
   
   Media: 95.1% ± 0.3%
   
   Modelo robusto ✓

Stratified K-Fold
-----------------

Mantiene proporción de clases:

.. code-block:: text

   Distribución preservada en cada fold
   Mejor para datasets desbalanceados

Manejo de Desbalanceo
=====================

Técnicas Disponibles
--------------------

**1. SMOTE (Synthetic Minority Over-sampling)**

.. code-block:: text

   Antes:
   Bajo: 6,074 (60.7%)
   Medio: 2,943 (29.4%)
   Alto: 983 (9.8%)
   
   Después de SMOTE:
   Bajo: 6,074 (33.3%)
   Medio: 6,074 (33.3%)
   Alto: 6,074 (33.3%)

**2. Class Weights**

.. code-block:: text

   Pesos automáticos:
   Bajo: 0.55
   Medio: 1.13
   Alto: 3.39

**3. Undersampling**

Reduce clase mayoritaria.

**4. Ensemble Methods**

Combina múltiples técnicas.

Guardar Modelos
===============

Guardar Mejor Modelo
---------------------

.. code-block:: text

   [💾 Guardar Modelo]
   
   Modelo: XGBoost
   Accuracy: 95.8%
   
   Guardado:
   ✓ Modelo entrenado
   ✓ Scaler
   ✓ Feature names
   ✓ Métricas
   ✓ Hiperparámetros
   
   Archivo: xgboost_best_20240115.pkl

Cargar Modelo
-------------

.. code-block:: text

   [📂 Cargar Modelo]
   
   ✓ Modelo cargado
   ✓ Listo para predicciones

Casos de Uso
============

**Caso 1: Producción**

Entrenar modelo final para sistema en vivo.

**Caso 2: Comparación**

Evaluar múltiples algoritmos para seleccionar el mejor.

**Caso 3: Investigación**

Experimentar con diferentes features y configuraciones.

Tips y Mejores Prácticas
=========================

✅ **Haz:**

- Entrena múltiples modelos
- Usa validación cruzada
- Optimiza hiperparámetros
- Maneja desbalanceo
- Guarda modelos y métricas

❌ **Evita:**

- Overfitting (validar siempre)
- Ignorar desbalanceo de clases
- No optimizar hiperparámetros
- Entrenar sin normalizar
- Usar solo accuracy como métrica

Troubleshooting
===============

**Problema: Overfitting**

Solución: Regularización, más datos, validación cruzada.

**Problema: Underfitting**

Solución: Modelo más complejo, más features, menos regularización.

**Problema: Entrenamiento lento**

Solución: LightGBM, menos datos, menos hiperparámetros.

Próximos Pasos
==============

Con tus modelos entrenados:

1. **Predecir**: :doc:`09_prediccion`
2. **Reentrenar**: :doc:`10_reentrenamiento`
3. **Aprender**: :doc:`11_rag_educativo`

¡Modelos listos para producción! 🤖