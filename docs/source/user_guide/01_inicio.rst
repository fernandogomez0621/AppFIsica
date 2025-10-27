=============================
1. Página de Inicio y Navegación
=============================

Esta guía te ayudará a familiarizarte con la interfaz de la aplicación y a navegar eficientemente entre los diferentes módulos.

Objetivo del Módulo
===================

La página de inicio es tu punto de partida en el sistema. Aquí podrás:

* 🏠 Ver un resumen general del sistema
* 📊 Acceder rápidamente a todos los módulos
* 📈 Visualizar el estado actual de tus datos y modelos
* 🎯 Entender el flujo de trabajo recomendado
* 🚀 Comenzar tu análisis de riesgo crediticio

Acceso a la Aplicación
======================

Iniciar la Aplicación
---------------------

**Paso 1: Activar el ambiente virtual**

.. code-block:: bash

   # En Linux/Mac
   source venv_fisica/bin/activate
   
   # En Windows
   venv_fisica\Scripts\activate

**Paso 2: Ejecutar Streamlit**

.. code-block:: bash

   streamlit run app.py

**Paso 3: Abrir en el navegador**

La aplicación se abrirá automáticamente en: ``http://localhost:8501``

.. note::
   Si el navegador no se abre automáticamente, copia y pega la URL en tu navegador preferido.

.. tip::
   Usa el script de activación rápida: ``./activate_env.sh`` (Linux/Mac)

Interfaz Principal
==================

Estructura de la Página
-----------------------

La interfaz está dividida en tres áreas principales:

1. **Sidebar (Barra lateral izquierda)**
   
   * Menú de navegación principal
   * Acceso a todos los módulos
   * Configuraciones globales

2. **Área de contenido central**
   
   * Dashboard principal
   * Visualizaciones y resultados
   * Formularios de entrada

3. **Barra superior**
   
   * Título de la aplicación
   * Menú de configuración de Streamlit
   * Opciones de tema (claro/oscuro)

Dashboard Principal
-------------------

Al abrir la aplicación, verás el dashboard principal con:

**Sección de Bienvenida:**

.. code-block:: text

   🏦 Sistema de Riesgo Crediticio con RBM
   =======================================
   
   Bienvenido al sistema integral de análisis y predicción
   de riesgo crediticio hipotecario.

**Tarjetas de Estado:**

Muestra el estado actual de:

* 📊 **Datos cargados**: Número de registros disponibles
* ⚡ **Modelos entrenados**: RBM y clasificadores activos
* 🎯 **Última predicción**: Timestamp de la última evaluación
* 📈 **Precisión del modelo**: Métricas de rendimiento

**Flujo de Trabajo Visual:**

Un diagrama interactivo que muestra:

1. Generar/Cargar Datos → 2. Validar → 3. Analizar → 4. Modelar → 5. Predecir

.. important::
   El dashboard se actualiza automáticamente cuando realizas cambios en cualquier módulo.

Navegación por Módulos
=======================

Sidebar - Menú Principal
------------------------

El sidebar contiene todos los módulos organizados por categoría:

**📊 Gestión de Datos:**

* **🏠 Inicio**: Dashboard principal (esta página)
* **📊 Generar Datos**: Crear datasets sintéticos
* **📁 Cargar Datos**: Importar datos externos

**📈 Análisis:**

* **📈 Análisis Descriptivo**: Estadísticas y visualizaciones
* **🔧 Ingeniería de Características**: Variables derivadas
* **🎯 Clustering**: Segmentación de clientes

**🤖 Modelado:**

* **⚡ Máquina de Boltzmann (RBM)**: Extracción de características
* **🤖 Modelos Supervisados**: Clasificadores de riesgo

**🔮 Predicción:**

* **🔮 Predicción de Riesgo**: Evaluar nuevos solicitantes
* **🔄 Reentrenamiento**: Actualizar modelos

**🎓 Educación:**

* **🎓 Aprende sobre RBMs**: Asistente RAG educativo

Cómo Navegar
------------

**Método 1: Click en el Sidebar**

1. Abre el sidebar (si está colapsado, click en ``>`` arriba a la izquierda)
2. Click en el módulo deseado
3. El contenido se carga en el área central

**Método 2: Atajos de Teclado**

* ``Ctrl + K`` (Windows/Linux) o ``Cmd + K`` (Mac): Abrir búsqueda rápida
* ``Ctrl + R`` (Windows/Linux) o ``Cmd + R`` (Mac): Recargar aplicación
* ``Esc``: Cerrar diálogos o modales

**Método 3: Flujo Secuencial**

Cada módulo incluye botones de navegación al final:

* **← Anterior**: Volver al módulo previo
* **Siguiente →**: Avanzar al siguiente paso

.. tip::
   Sigue el flujo secuencial si es tu primera vez usando el sistema.

Elementos de la Interfaz
=========================

Componentes Comunes
-------------------

**Botones de Acción:**

.. code-block:: text

   [🎯 Ejecutar Análisis]  [💾 Guardar Resultados]  [🔄 Reiniciar]

* **Primarios** (azul): Acciones principales
* **Secundarios** (gris): Acciones opcionales
* **Peligro** (rojo): Acciones destructivas (eliminar, reiniciar)

**Campos de Entrada:**

* **Numéricos**: Usa flechas o escribe directamente
* **Texto**: Escribe libremente
* **Selectores**: Despliega opciones con click
* **Sliders**: Arrastra para ajustar valores

**Expandibles:**

.. code-block:: text

   ▶ Configuración Avanzada
   
   (Click para expandir y ver más opciones)

**Pestañas:**

.. code-block:: text

   [Resumen] [Gráficos] [Datos] [Configuración]

Click en cada pestaña para cambiar de vista.

Mensajes y Notificaciones
--------------------------

**Tipos de Mensajes:**

.. note::
   **Información (azul)**: Consejos útiles y contexto adicional.

.. warning::
   **Advertencia (amarillo)**: Precauciones importantes.

.. error::
   **Error (rojo)**: Algo salió mal, revisa los detalles.

.. success::
   **Éxito (verde)**: Operación completada correctamente.

**Spinners de Carga:**

Cuando el sistema está procesando, verás:

.. code-block:: text

   ⏳ Procesando datos...
   ⏳ Entrenando modelo...
   ⏳ Generando visualizaciones...

.. tip::
   Los spinners indican que el sistema está trabajando. No recargues la página.

Estado del Sistema
==================

Indicadores de Estado
---------------------

**Panel de Estado (esquina superior derecha):**

* 🟢 **Verde**: Sistema operativo, todo funcionando
* 🟡 **Amarillo**: Advertencias menores, puede continuar
* 🔴 **Rojo**: Error crítico, requiere atención

**Información de Datos:**

.. code-block:: text

   📊 Dataset Actual
   ─────────────────
   Registros: 10,000
   Variables: 47
   Última actualización: 2024-01-15 10:30
   
   Distribución de Riesgo:
   • Bajo: 60.7% (6,074)
   • Medio: 29.4% (2,943)
   • Alto: 9.8% (983)

**Información de Modelos:**

.. code-block:: text

   🤖 Modelos Activos
   ──────────────────
   RBM: ✅ Entrenado (100 épocas)
   Random Forest: ✅ Entrenado (Acc: 94.2%)
   XGBoost: ✅ Entrenado (Acc: 95.1%)
   
   Última actualización: 2024-01-15 11:45

Configuración de la Aplicación
===============================

Menú de Configuración
---------------------

Click en el menú ``⋮`` (tres puntos) en la esquina superior derecha:

**Opciones Disponibles:**

* **Settings**: Configuración de Streamlit
* **Print**: Imprimir página actual
* **Record a screencast**: Grabar sesión
* **Report a bug**: Reportar problemas
* **Get help**: Documentación de Streamlit
* **About**: Información de la aplicación

Tema de la Aplicación
---------------------

**Cambiar entre tema claro y oscuro:**

1. Click en ``⋮`` → **Settings**
2. En **Theme**, selecciona:
   * **Light**: Tema claro (fondo blanco)
   * **Dark**: Tema oscuro (fondo negro)
   * **Use system setting**: Usar configuración del sistema

.. tip::
   El tema oscuro es más cómodo para sesiones largas y reduce la fatiga visual.

Configuración de Ejecución
---------------------------

**Always rerun:**

* ✅ Activado: La app se recarga automáticamente al cambiar código
* ❌ Desactivado: Debes recargar manualmente

**Run on save:**

* ✅ Activado: Cambios en archivos recargan la app
* ❌ Desactivado: Cambios no afectan la app en ejecución

.. note::
   Estas opciones son útiles para desarrolladores. Como usuario, déjalas en sus valores por defecto.

Casos de Uso Comunes
=====================

Caso 1: Primera Vez en el Sistema
----------------------------------

**Objetivo**: Familiarizarte con la interfaz y generar tu primer análisis.

**Pasos:**

1. **Explora el Dashboard**
   
   * Lee la información de bienvenida
   * Revisa el flujo de trabajo visual
   * Identifica los módulos disponibles

2. **Genera Datos de Prueba**
   
   * Ve a **📊 Generar Datos**
   * Usa configuración por defecto (10,000 registros)
   * Click en **"Generar Dataset"**

3. **Visualiza los Datos**
   
   * Ve a **📈 Análisis Descriptivo**
   * Selecciona variables como ``salario_mensual`` y ``nivel_riesgo``
   * Explora las visualizaciones

4. **Aprende sobre RBMs**
   
   * Ve a **🎓 Aprende sobre RBMs**
   * Haz preguntas básicas sobre el sistema

.. tip::
   Dedica 15-20 minutos a explorar cada módulo sin presión. La familiaridad con la interfaz mejorará tu eficiencia.

Caso 2: Análisis Rápido de Datos Existentes
--------------------------------------------

**Objetivo**: Cargar tus datos y obtener insights rápidos.

**Pasos:**

1. **Carga tus Datos**
   
   * Ve a **📁 Cargar Datos**
   * Arrastra tu archivo CSV/Excel
   * Valida la calidad de los datos

2. **Análisis Exploratorio**
   
   * Ve a **📈 Análisis Descriptivo**
   * Revisa estadísticas univariadas
   * Identifica outliers y patrones

3. **Genera Reporte**
   
   * Descarga visualizaciones
   * Exporta estadísticas
   * Comparte con tu equipo

Caso 3: Flujo Completo de Modelado
-----------------------------------

**Objetivo**: Entrenar modelos y realizar predicciones.

**Pasos:**

1. **Preparación** → **📊 Generar Datos** o **📁 Cargar Datos**
2. **Análisis** → **📈 Análisis Descriptivo**
3. **Ingeniería** → **🔧 Ingeniería de Características**
4. **RBM** → **⚡ Máquina de Boltzmann**
5. **Clasificadores** → **🤖 Modelos Supervisados**
6. **Predicción** → **🔮 Predicción de Riesgo**

.. important::
   Sigue este flujo secuencialmente para mejores resultados. Cada paso construye sobre el anterior.

Tips y Mejores Prácticas
=========================

Navegación Eficiente
--------------------

✅ **Haz:**

* Usa el sidebar para navegación rápida
* Guarda tu trabajo frecuentemente
* Revisa los mensajes de estado
* Sigue el flujo de trabajo recomendado

❌ **Evita:**

* Recargar la página durante procesamiento
* Saltar pasos críticos del flujo
* Ignorar mensajes de advertencia
* Cerrar la app sin guardar

Organización del Trabajo
-------------------------

**Nombra tus Archivos Claramente:**

.. code-block:: text

   ✅ Bueno:
   - creditos_enero_2024.csv
   - modelo_rbm_v2_20240115.pkl
   - predicciones_nuevos_clientes.csv
   
   ❌ Malo:
   - datos.csv
   - modelo.pkl
   - output.csv

**Mantén un Log de Experimentos:**

Documenta:

* Fecha y hora del análisis
* Parámetros utilizados
* Resultados obtenidos
* Decisiones tomadas

.. tip::
   Usa la función de descarga de reportes para mantener un historial automático.

Rendimiento
-----------

**Para Datasets Grandes (>50K registros):**

* Usa muestreo para análisis exploratorio
* Aumenta el batch_size en RBM
* Considera usar menos épocas inicialmente
* Monitorea el uso de memoria

**Para Sesiones Largas:**

* Guarda modelos intermedios
* Exporta resultados periódicamente
* Reinicia la app si se vuelve lenta
* Cierra pestañas no utilizadas

Troubleshooting
===============

Problemas Comunes
-----------------

**Problema 1: La aplicación no carga**

.. code-block:: text

   Error: ModuleNotFoundError: No module named 'streamlit'

**Solución:**

.. code-block:: bash

   # Verifica que el ambiente virtual esté activado
   which python  # Debe mostrar ruta a venv_fisica
   
   # Reinstala dependencias si es necesario
   pip install -r requirements.txt

---

**Problema 2: Sidebar no aparece**

**Solución:**

* Click en ``>`` en la esquina superior izquierda
* Presiona ``Ctrl + Shift + R`` para recargar
* Verifica el ancho de tu ventana (debe ser >768px)

---

**Problema 3: Página en blanco después de navegar**

**Solución:**

1. Espera 5-10 segundos (puede estar cargando)
2. Revisa la consola del navegador (F12) por errores
3. Recarga la página (F5)
4. Reinicia la aplicación si persiste

---

**Problema 4: Cambios no se reflejan**

**Solución:**

* Verifica que hayas guardado los cambios
* Recarga la página (F5)
* Limpia la caché del navegador
* Reinicia la aplicación Streamlit

Errores de Datos
----------------

**Error: "No hay datos cargados"**

**Causa**: No has generado ni cargado datos.

**Solución**: Ve a **📊 Generar Datos** o **📁 Cargar Datos** primero.

---

**Error: "Formato de archivo no soportado"**

**Causa**: Archivo no es CSV, Excel o Parquet.

**Solución**: Convierte tu archivo a uno de los formatos soportados.

---

**Error: "Columnas requeridas faltantes"**

**Causa**: Tu dataset no tiene las columnas esperadas.

**Solución**: Revisa la documentación de variables requeridas en :doc:`02_generar_datos`.

Errores de Rendimiento
-----------------------

**Síntoma: Aplicación muy lenta**

**Posibles causas y soluciones:**

1. **Dataset muy grande**
   
   * Reduce el tamaño del dataset
   * Usa muestreo estratificado
   * Aumenta la RAM disponible

2. **Muchos modelos entrenados**
   
   * Elimina modelos antiguos
   * Usa solo los modelos necesarios
   * Reinicia la aplicación

3. **Navegador con muchas pestañas**
   
   * Cierra pestañas no utilizadas
   * Usa modo incógnito
   * Prueba otro navegador

.. warning::
   Si la aplicación se congela por más de 2 minutos, reiníciala con ``Ctrl + C`` en la terminal.

Obtener Ayuda
=============

Recursos Disponibles
--------------------

**Dentro de la Aplicación:**

* 🎓 **Sistema RAG**: Haz preguntas sobre RBMs y el sistema
* 📖 **Tooltips**: Pasa el cursor sobre ``ⓘ`` para ayuda contextual
* 💡 **Ejemplos**: Cada módulo incluye casos de uso

**Documentación:**

* 📚 **Manual de Usuario**: Este documento
* 🔧 **API Reference**: Documentación técnica
* 📝 **README**: Guía de instalación

**Comunidad:**

* 💬 **GitHub Issues**: Reporta bugs y solicita features
* 📧 **Email**: soporte@sistema-fisica.com
* 🌐 **Foro**: Discusiones y preguntas

Reportar Problemas
------------------

Al reportar un problema, incluye:

1. **Descripción del problema**: Qué intentabas hacer
2. **Pasos para reproducir**: Cómo llegaste al error
3. **Mensaje de error**: Copia el texto completo
4. **Contexto**: Sistema operativo, versión de Python, etc.
5. **Screenshots**: Capturas de pantalla si es relevante

.. tip::
   Usa el botón **"Report a bug"** en el menú ``⋮`` para reportar automáticamente con contexto del sistema.

Próximos Pasos
==============

Ahora que conoces la interfaz, estás listo para:

1. **Generar tu primer dataset**: :doc:`02_generar_datos`
2. **Cargar datos externos**: :doc:`03_cargar_datos`
3. **Explorar análisis descriptivo**: :doc:`04_analisis_descriptivo`

.. note::
   Te recomendamos seguir el orden de los módulos para una experiencia de aprendizaje óptima.

¡Disfruta explorando el sistema! 🚀