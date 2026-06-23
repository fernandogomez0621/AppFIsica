============================
3. Cargar Datos Externos
============================

Esta guía te enseñará a cargar, validar y preparar tus propios datos de crédito hipotecario para análisis y modelado.

Objetivo del Módulo
===================

El módulo de carga de datos te permite:

* 📁 **Importar datos externos** en múltiples formatos
* ✅ **Validar calidad** automáticamente
* 🔧 **Limpiar y preparar** datos para análisis
* 📊 **Visualizar problemas** de calidad
* 💾 **Guardar datos procesados** para uso posterior
* 🔍 **Detectar anomalías** y valores atípicos

Formatos Soportados
===================

El sistema acepta los siguientes formatos:

**CSV (Comma-Separated Values)**

.. code-block:: text

   ✅ Extensión: .csv
   ✅ Encoding: UTF-8, Latin-1
   ✅ Separadores: coma, punto y coma, tabulador
   ✅ Tamaño máximo: 100 MB

**Excel**

.. code-block:: text

   ✅ Extensión: .xlsx, .xls
   ✅ Múltiples hojas soportadas
   ✅ Tamaño máximo: 50 MB

**Parquet**

.. code-block:: text

   ✅ Extensión: .parquet
   ✅ Formato columnar comprimido
   ✅ Ideal para datasets grandes
   ✅ Tamaño máximo: 200 MB

Acceso al Módulo
================

**Navegar al módulo:**

En el sidebar, click en:

.. code-block:: text

   📊 Gestión de Datos → 📁 Cargar Datos

Cargar Archivo
==============

Método 1: Arrastrar y Soltar
-----------------------------

1. Arrastra tu archivo desde el explorador
2. Suéltalo en el área designada
3. Espera la confirmación de carga

**Área de carga:**

.. code-block:: text

   ┌─────────────────────────────────────┐
   │  📁 Arrastra tu archivo aquí        │
   │                                     │
   │  o haz click para seleccionar      │
   │                                     │
   │  Formatos: CSV, Excel, Parquet     │
   │  Tamaño máximo: 100 MB             │
   └─────────────────────────────────────┘

Método 2: Seleccionar Archivo
------------------------------

1. Click en **"Browse files"**
2. Navega a tu archivo
3. Selecciona y abre

Validación Automática
=====================

Una vez cargado, el sistema ejecuta validaciones automáticas:

**Validación 1: Estructura del Archivo**

.. code-block:: text

   ✓ Archivo leído correctamente
   ✓ 10,000 registros detectados
   ✓ 25 columnas encontradas
   ✓ Encoding: UTF-8

**Validación 2: Columnas Requeridas**

El sistema verifica que existan las columnas mínimas:

* ``edad``
* ``salario_mensual``
* ``puntaje_datacredito``
* ``valor_inmueble``
* ``monto_credito``
* ``nivel_riesgo`` (opcional)

.. warning::
   Si faltan columnas críticas, el sistema te pedirá mapearlas o agregarlas.

**Validación 3: Tipos de Datos**

.. code-block:: text

   ✓ Variables numéricas: 18
   ✓ Variables categóricas: 7
   ✓ Conversiones necesarias: 2

**Validación 4: Valores Faltantes**

.. code-block:: text

   ⚠ Valores faltantes detectados:
   
   salario_mensual: 15 (0.15%)
   puntaje_datacredito: 8 (0.08%)
   ciudad: 3 (0.03%)

**Validación 5: Valores Atípicos**

.. code-block:: text

   ⚠ Outliers detectados:
   
   salario_mensual: 23 valores extremos
   valor_inmueble: 12 valores extremos

Reporte de Calidad
==================

El sistema genera un reporte completo:

**Resumen General:**

.. code-block:: text

   📊 REPORTE DE CALIDAD DE DATOS
   ================================
   
   Total de registros: 10,000
   Total de columnas: 25
   Completitud: 99.7%
   Calidad general: ★★★★☆ (4/5)

**Problemas Detectados:**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Problema
     - Severidad
     - Registros Afectados
   * - Valores faltantes
     - Media
     - 26 (0.26%)
   * - Outliers extremos
     - Baja
     - 35 (0.35%)
   * - Duplicados
     - Alta
     - 0 (0%)
   * - Inconsistencias
     - Media
     - 12 (0.12%)

Limpieza de Datos
=================

El módulo ofrece opciones de limpieza automática:

**Opción 1: Valores Faltantes**

.. code-block:: text

   Estrategia para valores faltantes:
   
   ○ Eliminar registros con faltantes
   ● Imputar con mediana (numéricos)
   ○ Imputar con moda (categóricos)
   ○ Dejar sin cambios

**Opción 2: Outliers**

.. code-block:: text

   Tratamiento de outliers:
   
   ○ Eliminar outliers extremos
   ● Winsorizar (reemplazar con percentiles)
   ○ Transformar (log, sqrt)
   ○ Dejar sin cambios

**Opción 3: Duplicados**

.. code-block:: text

   ☑ Eliminar registros duplicados

**Aplicar Limpieza:**

.. code-block:: text

   [🔧 Aplicar Limpieza Automática]

Resultados Después de Limpieza
-------------------------------

.. code-block:: text

   ✓ Limpieza completada
   
   Registros originales: 10,000
   Registros eliminados: 26
   Registros finales: 9,974
   
   Valores imputados:
   - salario_mensual: 15
   - puntaje_datacredito: 8
   - ciudad: 3
   
   Outliers tratados: 35

Mapeo de Columnas
=================

Si tus columnas tienen nombres diferentes, usa el mapeador:

**Ejemplo:**

.. code-block:: text

   Tus Columnas          →    Columnas del Sistema
   ─────────────────────────────────────────────────
   age                   →    edad
   monthly_salary        →    salario_mensual
   credit_score          →    puntaje_datacredito
   property_value        →    valor_inmueble
   loan_amount           →    monto_credito

.. tip::
   El sistema intenta detectar automáticamente las correspondencias.

Guardar Datos Procesados
=========================

Una vez validados y limpiados:

.. code-block:: text

   [💾 Guardar Datos Procesados]

Opciones de guardado:

* **En memoria**: Para uso inmediato en la sesión
* **CSV**: Para exportar y compartir
* **Parquet**: Para almacenamiento eficiente

Casos de Uso
============

**Caso 1: Datos de Producción**

Cargar datos reales de tu banco para análisis.

**Caso 2: Datos de Terceros**

Importar datasets de competencias o investigación.

**Caso 3: Combinar con Sintéticos**

Mezclar datos reales con sintéticos para aumentar volumen.

Troubleshooting
===============

**Problema: "Archivo muy grande"**

**Solución:**

* Divide el archivo en partes más pequeñas
* Usa formato Parquet (más comprimido)
* Filtra datos antes de cargar

**Problema: "Encoding error"**

**Solución:**

* Guarda el archivo como UTF-8
* Usa Excel para convertir encoding
* Especifica encoding manualmente

**Problema: "Columnas no reconocidas"**

**Solución:**

* Usa el mapeador de columnas
* Renombra columnas en tu archivo
* Consulta la documentación de variables requeridas

Próximos Pasos
==============

Con tus datos cargados y validados:

1. **Analizar**: :doc:`04_analisis_descriptivo`
2. **Crear características**: :doc:`05_ingenieria_caracteristicas`
3. **Entrenar modelos**: :doc:`07_rbm`

¡Tus datos están listos para análisis! 🎉