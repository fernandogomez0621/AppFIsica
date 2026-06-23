====================
Manual de Usuario
====================

Bienvenido al manual de usuario del **Sistema de Riesgo Crediticio con RBM**. Esta guía te llevará paso a paso a través de todas las funcionalidades de la aplicación Streamlit.

.. note::
   Este manual está diseñado para usuarios no técnicos. Si buscas documentación técnica del código, consulta la :doc:`api_reference`.

Introducción
============

¿Qué es este sistema?
---------------------

Este sistema integral permite analizar y predecir el riesgo crediticio hipotecario utilizando técnicas avanzadas de Machine Learning, específicamente **Máquinas de Boltzmann Restringidas (RBM)** combinadas con modelos supervisados.

**Características principales:**

* 📊 Generación de datos sintéticos realistas
* 📁 Carga y validación de datos externos
* 📈 Análisis exploratorio completo
* 🔧 Ingeniería de características automática
* ⚡ Entrenamiento de RBM para extracción de características
* 🤖 Múltiples modelos de clasificación supervisados
* 🔮 Sistema de predicción en tiempo real
* 🔄 Reentrenamiento automático de modelos
* 🎓 Asistente educativo RAG con papers científicos

¿Para quién es este sistema?
-----------------------------

**Analistas de Riesgo Crediticio:**
   Evalúa solicitudes de crédito hipotecario, identifica patrones de riesgo y optimiza políticas de aprobación.

**Estudiantes de Física:**
   Aprende sobre Máquinas de Boltzmann y sus aplicaciones en finanzas mediante el sistema RAG educativo.

**Data Scientists:**
   Experimenta con modelos generativos, compara algoritmos de ML y construye pipelines completos.

**Gerentes de Crédito:**
   Toma decisiones informadas basadas en análisis predictivos y visualizaciones claras.

Flujo de Trabajo Recomendado
=============================

Para obtener los mejores resultados, sigue este flujo de trabajo:

.. image:: _static/workflow_diagram.png
   :alt: Flujo de trabajo del sistema
   :align: center
   :width: 80%

1. **📊 Generar o Cargar Datos**
   
   Comienza generando un dataset sintético o cargando tus propios datos de crédito.

2. **🔍 Validar y Limpiar**
   
   Asegura la calidad de los datos mediante validación automática y limpieza.

3. **📈 Análisis Exploratorio**
   
   Comprende las distribuciones, correlaciones y patrones en tus datos.

4. **🔧 Ingeniería de Características**
   
   Crea variables derivadas que mejoren el poder predictivo.

5. **⚡ Entrenar RBM**
   
   Extrae características latentes mediante Máquinas de Boltzmann.

6. **🤖 Entrenar Modelos Supervisados**
   
   Construye clasificadores de riesgo usando las características extraídas.

7. **🔮 Realizar Predicciones**
   
   Evalúa el riesgo de nuevos solicitantes en tiempo real.

8. **🔄 Reentrenar Modelos**
   
   Actualiza los modelos con nuevos datos para mantener su precisión.

9. **🎓 Aprender sobre RBMs**
   
   Usa el asistente RAG para profundizar en la teoría y aplicaciones.

Requisitos Previos
==================

Antes de comenzar, asegúrate de tener:

**Conocimientos Básicos:**

* Conceptos financieros básicos (crédito, tasa de interés, etc.)
* Familiaridad con navegadores web
* Comprensión básica de estadística (opcional pero útil)

**Requisitos Técnicos:**

* Navegador web moderno (Chrome, Firefox, Safari, Edge)
* Conexión a internet (para el sistema RAG)
* Archivos de datos en formato CSV, Excel o Parquet (si cargas datos propios)

**Configuración del Sistema:**

* Python 3.8 o superior instalado
* Ambiente virtual activado
* Dependencias instaladas (ver :doc:`installation`)

Convenciones de este Manual
============================

A lo largo de este manual, utilizamos las siguientes convenciones:

.. note::
   **Notas** proporcionan información adicional útil o consejos.

.. warning::
   **Advertencias** indican precauciones importantes que debes tener en cuenta.

.. tip::
   **Tips** ofrecen sugerencias para mejorar tu experiencia o resultados.

.. important::
   **Importante** resalta información crítica que no debes pasar por alto.

**Elementos de interfaz:**

* Los botones se muestran como: **"Generar Datos"**
* Los campos de entrada se muestran como: *Número de registros*
* Los menús se muestran como: → **Sidebar** → **Generar Datos**

**Código y comandos:**

.. code-block:: bash

   # Los comandos de terminal se muestran así
   streamlit run app.py

**Valores y parámetros:**

* Los valores numéricos se muestran como: ``100``
* Los nombres de variables se muestran como: ``nivel_riesgo``
* Los archivos se muestran como: :file:`datos_credito.csv`

Estructura del Manual
=====================

Este manual está organizado en módulos que corresponden a las funcionalidades de la aplicación:

.. toctree::
   :maxdepth: 2
   :caption: Guías por Módulo:

   user_guide/01_inicio
   user_guide/02_generar_datos
   user_guide/03_cargar_datos
   user_guide/04_analisis_descriptivo
   user_guide/05_ingenieria_caracteristicas
   user_guide/06_clustering
   user_guide/07_rbm
   user_guide/08_modelos_supervisados
   user_guide/09_prediccion
   user_guide/10_reentrenamiento
   user_guide/11_rag_educativo

Cada guía incluye:

* **Objetivo del módulo**: Qué puedes lograr
* **Paso a paso**: Instrucciones detalladas
* **Parámetros configurables**: Qué significa cada opción
* **Interpretación de resultados**: Cómo leer las salidas
* **Casos de uso**: Ejemplos prácticos
* **Tips y mejores prácticas**: Recomendaciones expertas
* **Troubleshooting**: Solución de problemas comunes

Acceso Rápido
=============

**Primeros Pasos:**

* :doc:`user_guide/01_inicio` - Familiarízate con la interfaz
* :doc:`user_guide/02_generar_datos` - Crea tu primer dataset

**Análisis de Datos:**

* :doc:`user_guide/03_cargar_datos` - Trabaja con tus propios datos
* :doc:`user_guide/04_analisis_descriptivo` - Explora y visualiza

**Modelado Avanzado:**

* :doc:`user_guide/07_rbm` - Entrena Máquinas de Boltzmann
* :doc:`user_guide/08_modelos_supervisados` - Construye clasificadores

**Producción:**

* :doc:`user_guide/09_prediccion` - Predice riesgo en tiempo real
* :doc:`user_guide/10_reentrenamiento` - Mantén modelos actualizados

**Aprendizaje:**

* :doc:`user_guide/11_rag_educativo` - Aprende sobre RBMs con IA

Soporte y Recursos
==================

Si necesitas ayuda adicional:

**Documentación Técnica:**
   Consulta la :doc:`api_reference` para detalles de implementación.

**Instalación:**
   Revisa la guía de :doc:`installation` si tienes problemas de configuración.

**Preguntas Frecuentes:**
   Cada módulo incluye una sección de troubleshooting con soluciones a problemas comunes.

**Sistema RAG:**
   Usa el módulo educativo para hacer preguntas específicas sobre RBMs y el sistema.

**Comunidad:**
   Únete a las discusiones en GitHub Issues para compartir experiencias y obtener ayuda.

Próximos Pasos
==============

¡Estás listo para comenzar! Te recomendamos:

1. **Lee la guía de inicio**: :doc:`user_guide/01_inicio`
2. **Genera tu primer dataset**: :doc:`user_guide/02_generar_datos`
3. **Explora los datos**: :doc:`user_guide/04_analisis_descriptivo`
4. **Experimenta con RBM**: :doc:`user_guide/07_rbm`

.. tip::
   Si eres nuevo en Machine Learning, comienza con el módulo educativo RAG para aprender los conceptos fundamentales antes de entrenar modelos.

¡Disfruta explorando el sistema de riesgo crediticio con RBM! 🚀