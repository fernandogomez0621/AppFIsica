====================================
11. Sistema RAG Educativo
====================================

Esta guía te enseñará a usar el asistente educativo basado en RAG (Retrieval-Augmented Generation) para aprender sobre Máquinas de Boltzmann Restringidas y conceptos relacionados.

Objetivo del Módulo
===================

El sistema RAG educativo te permite:

* 🎓 **Aprender sobre RBMs** de forma interactiva
* 📚 **Consultar papers científicos** automáticamente
* 💬 **Chat con IA** especializada en física estadística
* 🔍 **Búsqueda semántica** en literatura científica
* 📖 **Citas y referencias** automáticas
* 📤 **Cargar tus propios PDFs** para análisis

¿Qué es RAG?
============

Definición
----------

**RAG (Retrieval-Augmented Generation)** es un sistema que:

1. **Recupera** información relevante de una base de conocimiento
2. **Aumenta** el contexto del LLM con esa información
3. **Genera** respuestas basadas en fuentes confiables

Arquitectura
------------

.. code-block:: text

   Tu Pregunta
       ↓
   [Embedding] → Búsqueda Vectorial
       ↓
   Base de Conocimiento (ChromaDB)
       ↓
   Top-K Documentos Relevantes
       ↓
   [LLM: Llama 3.3 70B] + Contexto
       ↓
   Respuesta con Citas

Acceso al Módulo
================

En el sidebar, click en:

.. code-block:: text

   🎓 Educación → 🎓 Aprende sobre RBMs

Base de Conocimiento
====================

Papers Incluidos
----------------

El sistema incluye 13 papers científicos fundamentales:

**Fundamentos de RBM:**

1. **Hinton (2002)** - Training Products of Experts by Minimizing Contrastive Divergence
2. **Hinton (2010)** - A Practical Guide to Training Restricted Boltzmann Machines
3. **Fischer & Igel (2012)** - An Introduction to Restricted Boltzmann Machines

**Deep Learning:**

4. **Hinton & Salakhutdinov (2006)** - Reducing the Dimensionality of Data with Neural Networks
5. **Salakhutdinov (2009)** - Deep Boltzmann Machines
6. **Bengio (2013)** - Better Mixing via Deep Representations

**Aplicaciones:**

7. **Salakhutdinov (2007)** - Restricted Boltzmann Machines for Collaborative Filtering
8. **Larochelle (2008)** - Classification using Discriminative RBMs

**Modelos Relacionados:**

9. **Ackley (1985)** - A Learning Algorithm for Boltzmann Machines
10. **Montúfar (2011)** - Universal Approximation Depth and Errors
11. **Goodfellow (2014)** - Generative Adversarial Networks
12. **Kingma (2014)** - Auto-Encoding Variational Bayes

Cargar Nuevos Papers
--------------------

.. code-block:: text

   [📤 Cargar Nuevo PDF]
   
   Formatos: PDF
   Tamaño máximo: 50 MB
   
   Archivo: mi_paper_rbm.pdf
   
   [⬆️ Subir y Procesar]
   
   Procesando...
   ✓ PDF extraído (45 páginas)
   ✓ Texto dividido en chunks
   ✓ Embeddings generados
   ✓ Agregado a base de conocimiento
   
   Listo para consultar ✓

Usar el Chat Educativo
======================

Interfaz del Chat
-----------------

.. code-block:: text

   ╔══════════════════════════════════════╗
   ║   🎓 Asistente Educativo RBM        ║
   ╠══════════════════════════════════════╣
   ║                                      ║
   ║  Haz preguntas sobre:                ║
   ║  • Máquinas de Boltzmann             ║
   ║  • Contrastive Divergence            ║
   ║  • Deep Learning                     ║
   ║  • Aplicaciones en finanzas          ║
   ║                                      ║
   ║  Escribe tu pregunta abajo...        ║
   ║                                      ║
   ╚══════════════════════════════════════╝
   
   Tu pregunta: _

Hacer una Pregunta
------------------

**Ejemplo 1: Pregunta Básica**

.. code-block:: text

   Tú: ¿Qué es una Máquina de Boltzmann Restringida?
   
   🤖 Asistente:
   
   Una Máquina de Boltzmann Restringida (RBM) es un modelo 
   generativo estocástico que consiste en dos capas de 
   unidades: una capa visible y una capa oculta. La 
   "restricción" significa que no hay conexiones dentro de 
   cada capa, solo entre capas.
   
   Características clave:
   
   1. **Arquitectura bipartita**: Unidades visibles (v) y 
      ocultas (h) sin conexiones intra-capa
   
   2. **Función de energía**: 
      E(v,h) = -∑ᵢ aᵢvᵢ - ∑ⱼ bⱼhⱼ - ∑ᵢⱼ vᵢWᵢⱼhⱼ
   
   3. **Entrenamiento**: Usa Contrastive Divergence (CD-k)
      para aproximar el gradiente de log-verosimilitud
   
   📚 Fuentes:
   • Hinton (2010) - A Practical Guide to Training RBMs
   • Fischer & Igel (2012) - Introduction to RBMs
   
   ¿Quieres saber más sobre algún aspecto específico?

**Ejemplo 2: Pregunta Técnica**

.. code-block:: text

   Tú: ¿Cómo funciona Contrastive Divergence?
   
   🤖 Asistente:
   
   Contrastive Divergence (CD-k) es un algoritmo de 
   aprendizaje para RBMs propuesto por Hinton (2002). 
   Aproxima el gradiente de la log-verosimilitud mediante 
   un proceso de muestreo truncado.
   
   **Algoritmo CD-1:**
   
   1. Inicializar v⁰ con datos de entrenamiento
   2. Calcular p(h|v⁰) y muestrear h⁰
   3. Calcular p(v|h⁰) y muestrear v¹
   4. Calcular p(h|v¹) para obtener h¹
   5. Actualizar pesos: ΔW ∝ ⟨v⁰h⁰⟩ - ⟨v¹h¹⟩
   
   **Ventajas:**
   • Mucho más rápido que MCMC completo
   • Funciona bien en práctica (k=1 suele ser suficiente)
   • Permite entrenamiento de redes profundas
   
   **Limitaciones:**
   • Es una aproximación (no gradiente exacto)
   • Puede tener problemas con modos múltiples
   
   📚 Fuentes:
   • Hinton (2002) - Training Products of Experts
   • Bengio (2013) - Better Mixing via Deep Representations
   
   ¿Te gustaría ver un ejemplo de implementación?

**Ejemplo 3: Aplicación Práctica**

.. code-block:: text

   Tú: ¿Cómo puedo usar RBM para riesgo crediticio?
   
   🤖 Asistente:
   
   Las RBMs son excelentes para riesgo crediticio por 
   varias razones:
   
   **1. Extracción de Características No Lineales:**
   
   Las RBMs pueden aprender representaciones latentes que 
   capturan patrones complejos en datos financieros que 
   modelos lineales no detectan.
   
   **2. Reducción de Dimensionalidad:**
   
   Si tienes 50 variables financieras, una RBM puede 
   comprimirlas a 20-30 características latentes más 
   informativas.
   
   **3. Detección de Anomalías:**
   
   El error de reconstrucción de la RBM puede identificar 
   solicitudes atípicas o fraudulentas.
   
   **Pipeline Recomendado:**
   
   1. Pre-entrenar RBM con datos no etiquetados
   2. Extraer activaciones de capa oculta
   3. Usar como features para Random Forest/XGBoost
   4. Típicamente mejora 3-5% en accuracy
   
   **Ejemplo de Código:**
   
   ```python
   # Entrenar RBM
   rbm = RBM(n_hidden=100, learning_rate=0.01)
   rbm.fit(X_train)
   
   # Extraer features
   X_train_rbm = rbm.transform(X_train)
   X_test_rbm = rbm.transform(X_test)
   
   # Entrenar clasificador
   clf = RandomForestClassifier()
   clf.fit(X_train_rbm, y_train)
   ```
   
   📚 Fuentes:
   • Salakhutdinov (2007) - Collaborative Filtering
   • Larochelle (2008) - Discriminative RBMs
   
   ¿Necesitas ayuda con la implementación?

Preguntas Sugeridas
===================

Conceptos Básicos
-----------------

.. code-block:: text

   💡 Preguntas Sugeridas:
   
   • ¿Qué es una Máquina de Boltzmann?
   • ¿Cuál es la diferencia entre RBM y Autoencoder?
   • ¿Qué es la función de energía en RBMs?
   • ¿Cómo se calculan las probabilidades en RBMs?
   • ¿Qué significa "restringida"?

Entrenamiento
-------------

.. code-block:: text

   💡 Preguntas Sugeridas:
   
   • ¿Cómo entrenar una RBM?
   • ¿Qué es Contrastive Divergence?
   • ¿Cuántas épocas necesito?
   • ¿Cómo elegir el learning rate?
   • ¿Qué es Gibbs sampling?

Aplicaciones
------------

.. code-block:: text

   💡 Preguntas Sugeridas:
   
   • ¿Para qué sirven las RBMs?
   • ¿Cómo usar RBM en clasificación?
   • ¿RBM vs PCA para reducción dimensional?
   • ¿Cómo detectar anomalías con RBM?
   • ¿RBM en sistemas de recomendación?

Avanzado
--------

.. code-block:: text

   💡 Preguntas Sugeridas:
   
   • ¿Qué son las Deep Boltzmann Machines?
   • ¿Cómo apilar múltiples RBMs?
   • ¿Qué es persistent Contrastive Divergence?
   • ¿Cómo regularizar RBMs?
   • ¿RBM vs VAE vs GAN?

Características del Sistema
============================

Búsqueda Semántica
------------------

El sistema usa embeddings para encontrar información relevante:

.. code-block:: text

   Tu pregunta: "problemas de convergencia en RBMs"
   
   Búsqueda semántica encuentra:
   1. Sección sobre "training difficulties" (relevancia: 0.89)
   2. Párrafo sobre "divergence issues" (relevancia: 0.85)
   3. Discusión de "learning rate tuning" (relevancia: 0.82)
   
   Contexto enviado al LLM con estos fragmentos

Citas Automáticas
-----------------

Todas las respuestas incluyen referencias:

.. code-block:: text

   📚 Fuentes citadas:
   
   [1] Hinton, G. E. (2002). Training products of experts 
       by minimizing contrastive divergence. Neural 
       computation, 14(8), 1771-1800.
   
   [2] Fischer, A., & Igel, C. (2012). An introduction to 
       restricted Boltzmann machines. In Progress in 
       Pattern Recognition (pp. 14-36).

Multilingüe
-----------

El sistema responde en el idioma de tu pregunta:

.. code-block:: text

   Pregunta en español → Respuesta en español
   Question in English → Answer in English

Historial de Conversación
--------------------------

El chat mantiene contexto:

.. code-block:: text

   Tú: ¿Qué es una RBM?
   🤖: [Explicación detallada]
   
   Tú: ¿Y cómo se entrena?
   🤖: [Explica entrenamiento, recordando que ya 
        explicó qué es una RBM]

Configuración Avanzada
======================

Parámetros del Sistema
----------------------

.. code-block:: text

   ⚙️ Configuración RAG:
   
   Modelo LLM: Llama 3.3 70B (Groq)
   Temperatura: 0.3 (más determinista)
   Top-K documentos: 6
   Chunk size: 1500 caracteres
   Chunk overlap: 300 caracteres
   
   [💾 Guardar Configuración]

Ajustar Temperatura
-------------------

.. code-block:: text

   Temperatura: [0.3]
   Rango: 0.0 - 1.0
   
   • 0.0-0.3: Respuestas precisas y consistentes
   • 0.4-0.6: Balance creatividad/precisión
   • 0.7-1.0: Respuestas más creativas

Casos de Uso
============

**Caso 1: Estudiante de Física**

Aprender conceptos de física estadística aplicados a ML.

**Caso 2: Data Scientist**

Entender implementación técnica de RBMs.

**Caso 3: Investigador**

Consultar literatura científica rápidamente.

**Caso 4: Desarrollador**

Obtener ejemplos de código y mejores prácticas.

Tips y Mejores Prácticas
=========================

✅ **Haz:**

- Preguntas específicas y claras
- Pide ejemplos de código
- Solicita referencias adicionales
- Usa el contexto de conversaciones previas

❌ **Evita:**

- Preguntas muy generales
- Esperar respuestas fuera del dominio (RBMs/ML)
- Ignorar las citas y referencias
- Preguntas sin contexto

Limitaciones
============

**El sistema NO puede:**

* Ejecutar código
* Acceder a internet en tiempo real
* Responder sobre temas fuera de los papers cargados
* Garantizar 100% de precisión (siempre verifica)

**El sistema SÍ puede:**

* Explicar conceptos de RBMs y ML
* Citar papers científicos
* Proporcionar ejemplos de código
* Responder preguntas técnicas detalladas

Troubleshooting
===============

**Problema: Respuesta no relevante**

Solución: Reformula la pregunta más específicamente.

**Problema: Sin citas**

Solución: Pregunta explícitamente por referencias.

**Problema: Respuesta muy técnica**

Solución: Pide una explicación más simple.

**Problema: Error de API**

Solución: Verifica API key de Groq en configuración.

Próximos Pasos
==============

Después de aprender:

1. **Aplicar conocimiento**: :doc:`07_rbm`
2. **Experimentar**: :doc:`08_modelos_supervisados`
3. **Compartir**: Documenta tus hallazgos

¡Disfruta aprendiendo sobre RBMs! 🎓