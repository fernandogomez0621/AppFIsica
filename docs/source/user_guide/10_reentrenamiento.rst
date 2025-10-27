====================================
10. Reentrenamiento de Modelos
====================================

Esta guía te enseñará a actualizar y reentrenar modelos con nuevos datos para mantener su precisión y relevancia en producción.

Objetivo del Módulo
===================

El módulo de reentrenamiento te permite:

* 🔄 **Actualizar modelos** con nuevos datos
* 📊 **Monitorear degradación** de rendimiento
* 🎯 **Detectar drift** de datos
* 📈 **Comparar versiones** de modelos
* 💾 **Versionar modelos** automáticamente
* 🔍 **Validar mejoras** antes de desplegar

Acceso al Módulo
================

En el sidebar, click en:

.. code-block:: text

   🔮 Predicción → 🔄 Reentrenamiento

¿Por Qué Reentrenar?
====================

Razones para Reentrenar
-----------------------

**1. Drift de Datos**

.. code-block:: text

   Distribución de salarios cambió:
   
   2023: Media = $4.2M
   2024: Media = $4.8M (+14%)
   
   Modelo desactualizado ⚠️

**2. Degradación de Rendimiento**

.. code-block:: text

   Accuracy en producción:
   
   Mes 1: 95.2%
   Mes 2: 94.8%
   Mes 3: 93.5%
   Mes 4: 91.2% ⚠️
   
   Reentrenamiento necesario

**3. Nuevos Datos Disponibles**

.. code-block:: text

   Datos de entrenamiento:
   Original: 10,000 registros
   Nuevos: 5,000 registros
   
   Total: 15,000 registros
   
   Oportunidad de mejorar

**4. Cambios en el Negocio**

* Nuevas políticas de crédito
* Cambios regulatorios
* Nuevos productos

Monitoreo de Modelos
====================

Dashboard de Rendimiento
-------------------------

.. code-block:: text

   Modelo: XGBoost v2.1
   Desplegado: 2024-01-01
   Días en producción: 45
   
   Métricas Actuales:
   • Accuracy: 92.1% (↓ 3.1% vs baseline)
   • Precision: 91.5% (↓ 3.3%)
   • Recall: 92.8% (↓ 2.4%)
   • F1-Score: 92.1% (↓ 2.9%)
   
   Estado: ⚠️ Degradación detectada

Detección de Drift
------------------

**Data Drift:**

.. code-block:: text

   Variables con drift significativo:
   
   • salario_mensual: KS-test p=0.001 ⚠️
   • valor_inmueble: KS-test p=0.003 ⚠️
   • puntaje_datacredito: KS-test p=0.045 ⚠️
   
   Acción: Reentrenamiento recomendado

**Concept Drift:**

.. code-block:: text

   Relación DTI → Riesgo cambió:
   
   Antes: Correlación = 0.68
   Ahora: Correlación = 0.54
   
   Modelo desactualizado

Alertas Automáticas
-------------------

.. code-block:: text

   🔔 Alertas Activas:
   
   1. Accuracy < 93% por 7 días consecutivos
   2. Data drift detectado en 3 variables
   3. 150 predicciones con baja confianza esta semana
   
   Recomendación: Reentrenar inmediatamente

Proceso de Reentrenamiento
===========================

Paso 1: Preparar Nuevos Datos
------------------------------

.. code-block:: text

   [📁 Cargar Nuevos Datos]
   
   Datos originales: 10,000
   Datos nuevos: 5,000
   
   Opciones:
   ○ Reemplazar datos antiguos
   ● Combinar con datos antiguos
   ○ Solo usar datos nuevos
   
   Total para reentrenamiento: 15,000

Paso 2: Validar Calidad
------------------------

.. code-block:: text

   [✅ Validar Nuevos Datos]
   
   ✓ Sin valores faltantes
   ✓ Distribuciones consistentes
   ✓ Sin duplicados
   ⚠ 12 outliers detectados (0.24%)
   
   Calidad: Buena

Paso 3: Configurar Reentrenamiento
-----------------------------------

.. code-block:: text

   Estrategia de reentrenamiento:
   
   ● Reentrenar desde cero
   ○ Fine-tuning (transfer learning)
   ○ Incremental learning
   
   Modelos a reentrenar:
   ☑ XGBoost
   ☑ LightGBM
   ☑ Random Forest
   ☐ RBM

Paso 4: Ejecutar Reentrenamiento
---------------------------------

.. code-block:: text

   [🔄 Iniciar Reentrenamiento]
   
   Reentrenando XGBoost...
   Época 1/100: Loss = 0.245
   Época 50/100: Loss = 0.089
   Época 100/100: Loss = 0.045
   ✓ Completado (3m 25s)
   
   Reentrenando LightGBM...
   ✓ Completado (2m 10s)
   
   Reentrenando Random Forest...
   ✓ Completado (4m 05s)
   
   Total: 9m 40s

Comparación de Versiones
=========================

Métricas Comparativas
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 20

   * - Modelo
     - Versión
     - Accuracy
     - F1-Score
     - Cambio
   * - XGBoost
     - v2.1 (old)
     - 92.1%
     - 92.1%
     - -
   * - XGBoost
     - v2.2 (new)
     - **95.8%**
     - **95.7%**
     - **+3.7%** ✓
   * - LightGBM
     - v1.5 (old)
     - 91.8%
     - 91.5%
     - -
   * - LightGBM
     - v1.6 (new)
     - **95.2%**
     - **95.0%**
     - **+3.4%** ✓

**Resultado:** Mejora significativa en todos los modelos

Tests A/B
---------

.. code-block:: text

   [🧪 Ejecutar Test A/B]
   
   Grupo A (v2.1): 1,000 predicciones
   Grupo B (v2.2): 1,000 predicciones
   
   Resultados:
   • Accuracy A: 92.3%
   • Accuracy B: 95.6%
   • Diferencia: +3.3% (p < 0.001)
   
   Conclusión: v2.2 es significativamente mejor

Validación Cruzada
------------------

.. code-block:: text

   5-Fold Cross-Validation:
   
   Modelo v2.1:
   Folds: [92.1%, 91.8%, 92.5%, 91.9%, 92.3%]
   Media: 92.1% ± 0.3%
   
   Modelo v2.2:
   Folds: [95.5%, 95.8%, 95.9%, 95.6%, 95.7%]
   Media: 95.7% ± 0.2%
   
   Mejora: +3.6% ✓

Versionado de Modelos
=====================

Sistema de Versiones
--------------------

.. code-block:: text

   Historial de Versiones:
   
   v2.2 (2024-02-15) ← Actual
   • Accuracy: 95.8%
   • Datos: 15,000
   • Cambios: Reentrenamiento completo
   
   v2.1 (2024-01-01)
   • Accuracy: 95.2%
   • Datos: 10,000
   • Cambios: Optimización hiperparámetros
   
   v2.0 (2023-12-01)
   • Accuracy: 94.1%
   • Datos: 10,000
   • Cambios: Versión inicial

Metadata de Versiones
----------------------

.. code-block:: text

   Modelo: XGBoost v2.2
   
   Información:
   • Fecha creación: 2024-02-15 10:30:00
   • Autor: sistema_auto
   • Datos entrenamiento: 15,000
   • Datos validación: 3,000
   • Tiempo entrenamiento: 3m 25s
   • Hiperparámetros: {...}
   • Features: 20
   • Métricas: {...}
   • Hash modelo: a3f5b2c...

Rollback
--------

.. code-block:: text

   [⏮️ Revertir a Versión Anterior]
   
   Versión actual: v2.2
   Revertir a: v2.1
   
   ⚠️ Advertencia:
   Perderás mejoras de +3.6% accuracy
   
   ¿Confirmar rollback? [Sí] [No]

Estrategias de Reentrenamiento
===============================

1. Reentrenamiento Completo
----------------------------

**Cuándo usar:**

* Drift significativo
* Muchos datos nuevos
* Cambios estructurales

**Ventajas:**

* Modelo completamente actualizado
* Mejor rendimiento

**Desventajas:**

* Más tiempo y recursos
* Pierde conocimiento previo

2. Fine-Tuning
--------------

**Cuándo usar:**

* Pocos datos nuevos
* Drift moderado
* Ajustes menores

**Ventajas:**

* Rápido
* Mantiene conocimiento previo

**Desventajas:**

* Mejora limitada

3. Incremental Learning
-----------------------

**Cuándo usar:**

* Datos continuos
* Actualizaciones frecuentes
* Recursos limitados

**Ventajas:**

* Muy eficiente
* Actualización continua

**Desventajas:**

* No todos los algoritmos lo soportan

Automatización
==============

Reentrenamiento Programado
---------------------------

.. code-block:: text

   [⏰ Configurar Reentrenamiento Automático]
   
   Frecuencia: ● Mensual ○ Semanal ○ Diario
   Día: 1 de cada mes
   Hora: 02:00 AM
   
   Condiciones:
   ☑ Solo si accuracy < 94%
   ☑ Solo si hay > 1,000 datos nuevos
   ☑ Notificar por email
   
   [💾 Guardar Configuración]

Triggers Automáticos
--------------------

.. code-block:: text

   Triggers configurados:
   
   1. Accuracy < 93% por 5 días
      → Reentrenar automáticamente
   
   2. Data drift detectado
      → Notificar y sugerir reentrenamiento
   
   3. 5,000 nuevos datos acumulados
      → Reentrenar automáticamente

Pipeline CI/CD
--------------

.. code-block:: text

   Pipeline de Reentrenamiento:
   
   1. Detectar trigger
   2. Validar nuevos datos
   3. Reentrenar modelos
   4. Validar mejora (>2%)
   5. Test A/B (1 semana)
   6. Desplegar si exitoso
   7. Monitorear rendimiento

Casos de Uso
============

**Caso 1: Reentrenamiento Mensual**

Actualización rutinaria con datos del mes.

**Caso 2: Reentrenamiento de Emergencia**

Degradación crítica detectada, reentrenar inmediatamente.

**Caso 3: Reentrenamiento por Cambio Regulatorio**

Nueva ley requiere actualizar criterios de riesgo.

Tips y Mejores Prácticas
=========================

✅ **Haz:**

- Monitorea rendimiento continuamente
- Versiona todos los modelos
- Valida mejoras antes de desplegar
- Mantén datos de entrenamiento históricos
- Documenta cambios

❌ **Evita:**

- Reentrenar sin validar mejora
- Desplegar sin test A/B
- Perder versiones anteriores
- Ignorar drift de datos
- Reentrenar demasiado frecuentemente

Troubleshooting
===============

**Problema: Modelo nuevo peor que anterior**

Solución: Rollback y revisar datos nuevos.

**Problema: Reentrenamiento muy lento**

Solución: Usa menos datos o algoritmo más rápido.

**Problema: Drift no detectado**

Solución: Ajusta umbrales de detección.

Próximos Pasos
==============

Después de reentrenar:

1. **Predecir**: :doc:`09_prediccion`
2. **Aprender**: :doc:`11_rag_educativo`
3. **Monitorear**: Continuar vigilando rendimiento

¡Modelos actualizados y optimizados! 🔄