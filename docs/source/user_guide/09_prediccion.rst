============================
9. Predicción de Riesgo
============================

Esta guía te enseñará a usar los modelos entrenados para predecir el nivel de riesgo crediticio de nuevos solicitantes en tiempo real.

Objetivo del Módulo
===================

El módulo de predicción te permite:

* 🔮 **Predecir riesgo** de nuevos solicitantes
* 📊 **Obtener probabilidades** por clase
* 🎯 **Evaluar confianza** de predicciones
* 📋 **Generar reportes** detallados
* 💾 **Guardar predicciones** para auditoría
* 🔄 **Predicción batch** para múltiples casos

Acceso al Módulo
================

En el sidebar, click en:

.. code-block:: text

   🔮 Predicción → 🔮 Predicción de Riesgo

Predicción Individual
=====================

Ingresar Datos del Solicitante
-------------------------------

**Información Personal:**

.. code-block:: text

   Edad: [35]
   Ciudad: [Bogotá ▼]
   Estrato: [4 ▼]
   Nivel educación: [Profesional ▼]
   Estado civil: [Casado ▼]
   Personas a cargo: [2]

**Información Laboral:**

.. code-block:: text

   Tipo empleo: [Formal ▼]
   Antigüedad (años): [5.5]
   Salario mensual: [$5,000,000]
   Egresos mensuales: [$3,200,000]

**Información Financiera:**

.. code-block:: text

   Puntaje DataCrédito: [750]
   Número demandas: [0]
   Número propiedades: [1]
   Patrimonio total: [$80,000,000]
   Saldo banco: [$15,000,000]

**Información del Crédito:**

.. code-block:: text

   Valor inmueble: [$250,000,000]
   Cuota inicial (%): [20]
   Plazo (años): [20]
   Tasa interés (%): [10.5]

Ejecutar Predicción
-------------------

.. code-block:: text

   [🎯 Predecir Nivel de Riesgo]
   
   Procesando...
   ✓ Datos validados
   ✓ Features calculadas
   ✓ Modelo aplicado

Resultado de la Predicción
===========================

Predicción Principal
--------------------

.. code-block:: text

   ╔══════════════════════════════════════╗
   ║   RESULTADO DE LA PREDICCIÓN        ║
   ╠══════════════════════════════════════╣
   ║                                      ║
   ║   Nivel de Riesgo: BAJO ✓           ║
   ║                                      ║
   ║   Confianza: 92.3%                   ║
   ║                                      ║
   ╚══════════════════════════════════════╝

Probabilidades por Clase
-------------------------

.. code-block:: text

   Distribución de Probabilidades:
   
   Bajo:  92.3% ████████████████████████████
   Medio:  6.5% ██
   Alto:   1.2% ▌

**Interpretación:**

* El modelo está 92.3% seguro de que es Bajo riesgo
* Probabilidad muy baja de ser Alto riesgo (1.2%)
* Predicción confiable

Factores Clave
--------------

.. code-block:: text

   Factores que Influyen en la Predicción:
   
   ✓ Positivos (reducen riesgo):
   1. Puntaje DataCrédito alto (750)
   2. DTI bajo (25.6%)
   3. Capacidad residual alta ($1,200,000)
   4. Empleo formal estable (5.5 años)
   5. Sin demandas legales
   
   ⚠ Negativos (aumentan riesgo):
   1. Ninguno significativo

Métricas Calculadas
-------------------

.. code-block:: text

   Ratios Financieros:
   
   • DTI: 25.6% (Excelente)
   • LTV: 80.0% (Aceptable)
   • Capacidad ahorro: $1,800,000
   • Capacidad residual: $1,200,000
   • Meses colchón: 11.5
   • Ratio patrimonio/deuda: 0.40

Recomendación
-------------

.. code-block:: text

   ╔══════════════════════════════════════╗
   ║         RECOMENDACIÓN                ║
   ╠══════════════════════════════════════╣
   ║                                      ║
   ║  ✅ APROBAR CRÉDITO                  ║
   ║                                      ║
   ║  Condiciones sugeridas:              ║
   ║  • Tasa: 10.5% (estándar)           ║
   ║  • Plazo: 20 años                    ║
   ║  • Cuota inicial: 20%                ║
   ║  • Seguimiento: Estándar             ║
   ║                                      ║
   ╚══════════════════════════════════════╝

Predicción Batch
================

Cargar Archivo
--------------

.. code-block:: text

   [📁 Cargar Archivo de Solicitantes]
   
   Formatos: CSV, Excel
   Columnas requeridas: edad, salario_mensual, ...
   
   Archivo cargado: nuevos_solicitantes.csv
   Registros: 150

Ejecutar Predicciones
---------------------

.. code-block:: text

   [🎯 Predecir Batch]
   
   Procesando 150 solicitantes...
   
   Progreso: ████████████████████ 100%
   
   ✓ Predicciones completadas
   Tiempo: 3.2 segundos

Resultados Batch
----------------

.. code-block:: text

   Resumen de Predicciones:
   
   Bajo:  92 (61.3%) ████████████████████
   Medio: 41 (27.3%) ████████
   Alto:  17 (11.3%) ███
   
   Recomendaciones:
   • Aprobar: 92 solicitudes
   • Revisar: 41 solicitudes
   • Rechazar: 17 solicitudes

Exportar Resultados
-------------------

.. code-block:: text

   [💾 Exportar Resultados]
   
   Formatos disponibles:
   ☑ CSV con predicciones
   ☑ Excel con formato
   ☑ PDF con reportes individuales
   ☑ JSON para API
   
   Archivo: predicciones_20240115.xlsx

Análisis de Confianza
=====================

Niveles de Confianza
--------------------

.. code-block:: text

   Distribución de Confianza:
   
   Alta (>90%):    105 (70.0%)
   Media (70-90%):  32 (21.3%)
   Baja (<70%):     13 ( 8.7%)

**Recomendación:**

* Alta confianza: Decisión automática
* Media confianza: Revisión rápida
* Baja confianza: Análisis manual detallado

Casos Límite
------------

Solicitantes cerca del umbral de decisión:

.. code-block:: text

   Casos Límite (requieren revisión):
   
   ID: 1023
   Predicción: Medio (48%) vs Bajo (47%)
   Acción: Revisar manualmente
   
   ID: 1087
   Predicción: Alto (52%) vs Medio (45%)
   Acción: Análisis detallado

Comparación de Modelos
======================

Predicción con Múltiples Modelos
---------------------------------

.. code-block:: text

   [🔄 Comparar Modelos]
   
   Solicitante ID: 1001
   
   XGBoost:      Bajo (92.3%)
   LightGBM:     Bajo (91.8%)
   Random Forest: Bajo (89.5%)
   SVM:          Bajo (87.2%)
   
   Consenso: BAJO ✓
   Confianza agregada: 90.2%

Ensemble Voting
---------------

.. code-block:: text

   Votación por mayoría:
   
   Bajo:  4 votos (XGBoost, LightGBM, RF, SVM)
   Medio: 0 votos
   Alto:  0 votos
   
   Decisión final: BAJO (unánime)

Monitoreo de Predicciones
==========================

Dashboard de Predicciones
--------------------------

.. code-block:: text

   Estadísticas del Día:
   
   Total predicciones: 247
   Aprobadas: 152 (61.5%)
   Revisión: 68 (27.5%)
   Rechazadas: 27 (10.9%)
   
   Tiempo promedio: 0.8s
   Confianza promedio: 88.3%

Alertas Automáticas
-------------------

.. code-block:: text

   ⚠ Alertas Activas:
   
   • 5 predicciones con baja confianza (<70%)
   • 2 casos con datos inconsistentes
   • 1 solicitante con múltiples demandas

Auditoría y Trazabilidad
=========================

Log de Predicciones
-------------------

Cada predicción se registra:

.. code-block:: text

   Timestamp: 2024-01-15 14:30:25
   ID Solicitante: SOL-2024-001
   Modelo: XGBoost v2.1
   Predicción: Bajo
   Probabilidad: 92.3%
   Usuario: analista@banco.com
   IP: 192.168.1.100

Explicabilidad
--------------

.. code-block:: text

   [📊 Explicar Predicción]
   
   SHAP Values:
   • puntaje_datacredito: +0.35
   • dti: +0.28
   • capacidad_residual: +0.22
   • ltv: -0.08
   • edad: +0.05

Casos de Uso
============

**Caso 1: Evaluación en Sucursal**

Analista evalúa solicitante en tiempo real.

**Caso 2: Procesamiento Nocturno**

Batch de 1,000 solicitudes procesadas automáticamente.

**Caso 3: API de Predicción**

Sistema externo consulta predicciones vía API.

**Caso 4: Revisión de Cartera**

Re-evaluar riesgo de clientes existentes.

Tips y Mejores Prácticas
=========================

✅ **Haz:**

- Valida datos de entrada
- Revisa casos de baja confianza
- Documenta decisiones
- Monitorea drift del modelo
- Actualiza modelos periódicamente

❌ **Evita:**

- Decisiones automáticas sin umbral de confianza
- Ignorar alertas del sistema
- No auditar predicciones
- Usar modelos desactualizados
- Omitir validación de datos

Troubleshooting
===============

**Problema: Predicción inconsistente**

Solución: Verifica calidad de datos de entrada.

**Problema: Baja confianza**

Solución: Revisa manualmente, puede ser caso atípico.

**Problema: Error en predicción**

Solución: Verifica que modelo esté cargado correctamente.

Próximos Pasos
==============

Después de predecir:

1. **Reentrenar**: :doc:`10_reentrenamiento`
2. **Aprender**: :doc:`11_rag_educativo`
3. **Analizar**: Volver a :doc:`04_analisis_descriptivo`

¡Sistema de predicción operativo! 🔮