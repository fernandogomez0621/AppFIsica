educational_rag
===============

.. automodule:: src.educational_rag
   :members:
   :undoc-members:
   :show-inheritance:

Descripción General
-------------------

Sistema RAG (Retrieval-Augmented Generation) educativo para aprender sobre Máquinas de Boltzmann Restringidas usando papers científicos y Groq AI. Combina búsqueda semántica con generación de respuestas.

Clases Principales
------------------

EducationalRAG
^^^^^^^^^^^^^^

.. autoclass:: src.educational_rag.EducationalRAG
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Descripción:**
   
   Sistema RAG educativo que combina ChromaDB para búsqueda vectorial y Groq AI para generación de respuestas.
   
   **Componentes:**
   
   * **ChromaDB:** Base de datos vectorial para papers científicos
   * **Groq AI:** LLM (Llama 3.3 70B) para generación de respuestas
   * **Papers científicos:** Colección de artículos sobre RBMs
   
   **Atributos:**
   
   * ``groq_client``: Cliente de Groq AI
   * ``chroma_client``: Cliente de ChromaDB
   * ``collection``: Colección de documentos
   * ``papers_dir``: Directorio de papers (articles/)
   
   **Ejemplo de uso:**
   
   .. code-block:: python
   
      from src.educational_rag import EducationalRAG
      
      # Inicializar sistema RAG
      rag = EducationalRAG()
      
      # Cargar papers a la base de datos
      rag.load_papers_to_db()
      
      # Hacer una pregunta
      question = "¿Qué es una Máquina de Boltzmann Restringida?"
      
      # Buscar contexto relevante
      context = rag.query_papers(question, n_results=3)
      
      # Generar respuesta
      response = rag.generate_response(question, context)
      
      print(f"Pregunta: {question}")
      print(f"\nRespuesta: {response}")

Métodos de Inicialización
--------------------------

_init_groq
^^^^^^^^^^

Inicializa el cliente de Groq AI.

**Configuración:**
   * Busca API key en ``st.secrets['GROQ_API_KEY']``
   * O en variable de entorno ``GROQ_API_KEY``
   * Modelo: Llama 3.3 70B Versatile

**Configurar API key:**

En ``.streamlit/secrets.toml``:

.. code-block:: toml

   GROQ_API_KEY = "gsk_..."

_init_chromadb
^^^^^^^^^^^^^^

Inicializa ChromaDB para almacenamiento vectorial.

**Configuración:**
   * Directorio: ``chroma_rbm_db/``
   * Colección: ``rbm_papers``
   * Modo: Persistente

Métodos de Gestión de Papers
-----------------------------

get_available_papers
^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.educational_rag.EducationalRAG.get_available_papers
   :noindex:

Obtiene lista de papers PDF disponibles.

**Returns:**
   Lista de nombres de archivos PDF

**Ejemplo:**

.. code-block:: python

   papers = rag.get_available_papers()
   
   print(f"Papers disponibles: {len(papers)}")
   for paper in papers:
       print(f"  - {paper}")

get_paper_references
^^^^^^^^^^^^^^^^^^^^

.. automethod:: src.educational_rag.EducationalRAG.get_paper_references
   :noindex:

Carga referencias bibliográficas desde JSON.

**Returns:**
   Diccionario con referencias completas

**Estructura de referencias:**

.. code-block:: python

   {
       "Hinton_2010_Practical_Guide.pdf": {
           "title": "A Practical Guide to Training RBMs",
           "authors": "Geoffrey E. Hinton",
           "year": 2010,
           "publication": "Technical Report UTML TR 2010-003",
           "citation": "Hinton, G. E. (2010). A Practical Guide..."
       }
   }

load_papers_to_db
^^^^^^^^^^^^^^^^^

.. automethod:: src.educational_rag.EducationalRAG.load_papers_to_db
   :noindex:

Carga papers a ChromaDB si no están cargados.

**Returns:**
   True si exitoso

**Proceso:**

1. Verificar si ya hay documentos en la colección
2. Si no hay PDFs, crear base de conocimiento básica
3. Indexar documentos con embeddings

_create_basic_knowledge
^^^^^^^^^^^^^^^^^^^^^^^

Crea base de conocimiento básica sobre RBMs.

**Temas cubiertos:**

1. **Introducción:** Qué es una RBM
2. **Arquitectura:** Estructura bipartita
3. **Contrastive Divergence:** Algoritmo de entrenamiento
4. **Función de energía:** Definición matemática
5. **Aplicaciones:** Usos prácticos
6. **Comparación:** RBM vs redes neuronales
7. **Proceso de entrenamiento:** Pasos detallados
8. **Limitaciones:** Desventajas conocidas
9. **Deep Belief Networks:** Apilamiento de RBMs
10. **PCD:** Persistent Contrastive Divergence

Métodos de Búsqueda
-------------------

query_papers
^^^^^^^^^^^^

.. automethod:: src.educational_rag.EducationalRAG.query_papers
   :noindex:

Busca información relevante en los papers usando similitud coseno.

**Parameters:**
   * ``question`` (str): Pregunta del usuario
   * ``n_results`` (int): Número de fragmentos a retornar (default: 3)

**Returns:**
   Lista de diccionarios con fragmentos relevantes

**Métrica de similitud:**
   * **Similitud Coseno:** Mide ángulo entre vectores de embeddings
   * Rango: [0, 1] donde 1 = idéntico

**Ejemplo:**

.. code-block:: python

   # Buscar información
   question = "¿Cómo funciona Contrastive Divergence?"
   context = rag.query_papers(question, n_results=3)
   
   # Ver resultados
   for i, doc in enumerate(context):
       print(f"\nFragmento {i+1}:")
       print(f"  Similitud: {doc['similarity']:.2%}")
       print(f"  Fuente: {doc['metadata']['source']}")
       print(f"  Autor: {doc['metadata']['author']}")
       print(f"  Contenido: {doc['content'][:200]}...")

Métodos de Generación
---------------------

generate_response
^^^^^^^^^^^^^^^^^

.. automethod:: src.educational_rag.EducationalRAG.generate_response
   :noindex:

Genera respuesta usando Groq AI basada en el contexto recuperado.

**Parameters:**
   * ``question`` (str): Pregunta del usuario
   * ``context`` (List[Dict]): Fragmentos relevantes de papers

**Returns:**
   Respuesta generada en lenguaje natural

**Configuración del LLM:**
   * Modelo: Llama 3.3 70B Versatile
   * Temperature: 0.7 (balance creatividad/precisión)
   * Max tokens: 2000

**Ejemplo:**

.. code-block:: python

   # Buscar contexto
   context = rag.query_papers("¿Qué es la función de energía en RBMs?", n_results=3)
   
   # Generar respuesta
   response = rag.generate_response(
       question="¿Qué es la función de energía en RBMs?",
       context=context
   )
   
   print(response)

_get_reference_mapping
^^^^^^^^^^^^^^^^^^^^^^

Obtiene mapeo de referencias bibliográficas.

**Returns:**
   Diccionario con información de citación

**Papers incluidos:**

* Hinton (2010): Practical Guide to Training RBMs
* Hinton (2002): Contrastive Divergence
* Hinton & Salakhutdinov (2006): Reducing Dimensionality
* Salakhutdinov et al. (2007): Collaborative Filtering
* Bengio (2013): Deep Learning of Representations
* Tieleman (2008): Persistent Contrastive Divergence

Funciones de Renderizado
-------------------------

render_educational_rag_module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.educational_rag.render_educational_rag_module

Renderiza la interfaz completa del sistema RAG en Streamlit.

**Funcionalidades:**

1. **Gestión de papers:**
   
   * Ver lista de papers disponibles
   * Descargar papers desde URL
   * Editar referencias bibliográficas
   * Eliminar papers

2. **Interfaz de chat:**
   
   * Preguntas sugeridas (Básico/Intermedio/Avanzado)
   * Campo de texto para preguntas personalizadas
   * Búsqueda con similitud coseno
   * Visualización de fragmentos relevantes
   * Respuestas generadas con IA

3. **Historial:**
   
   * Últimas 5 conversaciones
   * Preguntas y respuestas guardadas

Preguntas Sugeridas
-------------------

Nivel Básico
^^^^^^^^^^^^

* ¿Qué es una Máquina de Boltzmann Restringida?
* ¿Cuál es la diferencia entre RBM y una red neuronal tradicional?
* ¿Para qué se usan las RBMs?

Nivel Intermedio
^^^^^^^^^^^^^^^^

* ¿Cómo funciona el algoritmo Contrastive Divergence?
* ¿Qué son las unidades visibles y ocultas en una RBM?
* ¿Cómo se entrenan las RBMs?

Nivel Avanzado
^^^^^^^^^^^^^^

* ¿Cuáles son las limitaciones de las RBMs?
* ¿Cómo se apilan RBMs para crear Deep Belief Networks?
* ¿Qué es Persistent Contrastive Divergence?

Ejemplo Completo de Uso
------------------------

.. code-block:: python

   from src.educational_rag import EducationalRAG
   
   # Inicializar sistema
   rag = EducationalRAG()
   
   # Verificar configuración
   if not rag.groq_client:
       print("❌ Configura GROQ_API_KEY")
       exit()
   
   # Cargar papers
   success = rag.load_papers_to_db()
   if success:
       print(f"✅ Base de conocimiento lista")
       print(f"   Documentos: {rag.collection.count()}")
   
   # Lista de preguntas
   questions = [
       "¿Qué es una RBM?",
       "¿Cómo funciona Contrastive Divergence?",
       "¿Cuáles son las aplicaciones de las RBMs?",
       "¿Qué es la función de energía?",
       "¿Cómo se apilan RBMs en DBNs?"
   ]
   
   # Procesar cada pregunta
   for question in questions:
       print(f"\n{'='*60}")
       print(f"❓ {question}")
       print(f"{'='*60}")
       
       # Buscar contexto
       context = rag.query_papers(question, n_results=3)
       
       print(f"\n📚 Fragmentos encontrados: {len(context)}")
       for i, doc in enumerate(context):
           print(f"\n  Fragmento {i+1}:")
           print(f"    Similitud: {doc['similarity']:.2%}")
           print(f"    Fuente: {doc['metadata']['source']}")
           print(f"    Autor: {doc['metadata']['author']} ({doc['metadata']['year']})")
       
       # Generar respuesta
       response = rag.generate_response(question, context)
       
       print(f"\n💡 Respuesta:")
       print(f"{response}")
       print(f"\n{'='*60}")

Gestión de Papers Personalizados
---------------------------------

Agregar Paper desde URL
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import requests
   from pathlib import Path
   
   # Descargar paper
   url = "https://ejemplo.com/paper.pdf"
   response = requests.get(url)
   
   # Guardar
   papers_dir = Path("articles")
   papers_dir.mkdir(exist_ok=True)
   
   filepath = papers_dir / "nuevo_paper.pdf"
   with open(filepath, 'wb') as f:
       f.write(response.content)
   
   print(f"✅ Paper descargado: {filepath}")

Editar Referencias
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import json
   from pathlib import Path
   
   # Cargar referencias existentes
   ref_file = Path("articles/papers_references.json")
   
   if ref_file.exists():
       with open(ref_file, 'r') as f:
           references = json.load(f)
   else:
       references = {}
   
   # Agregar nueva referencia
   references["nuevo_paper.pdf"] = {
       "title": "Título del Paper",
       "authors": "Autor et al.",
       "year": 2024,
       "publication": "Journal Name, Vol(Issue), Pages",
       "citation": "Autor et al. (2024). Título del Paper. Journal Name."
   }
   
   # Guardar
   with open(ref_file, 'w', encoding='utf-8') as f:
       json.dump(references, f, indent=2, ensure_ascii=False)
   
   print("✅ Referencia agregada")

Arquitectura del Sistema RAG
-----------------------------

Flujo de Procesamiento
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   ┌─────────────────┐
   │  Pregunta del   │
   │     Usuario     │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │   ChromaDB      │
   │  Búsqueda de    │
   │   Similitud     │
   │   Coseno        │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Top-K Papers   │
   │   Relevantes    │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │   Groq AI       │
   │  (Llama 3.3)    │
   │  Generación     │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │   Respuesta     │
   │   Educativa     │
   └─────────────────┘

Embeddings y Similitud
^^^^^^^^^^^^^^^^^^^^^^

ChromaDB usa embeddings para representar documentos:

1. **Embedding:** Convierte texto a vector denso
2. **Similitud Coseno:** Mide ángulo entre vectores

.. math::

   \text{Similitud}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}

* Rango: [-1, 1], normalizado a [0, 1]
* 1.0 = Idéntico
* 0.0 = Completamente diferente

Métodos de Búsqueda
-------------------

query_papers
^^^^^^^^^^^^

.. automethod:: src.educational_rag.EducationalRAG.query_papers
   :noindex:

Busca fragmentos relevantes usando similitud vectorial.

**Parameters:**
   * ``question`` (str): Pregunta del usuario
   * ``n_results`` (int): Número de fragmentos (default: 3)

**Returns:**
   Lista de diccionarios con:
   
   * ``content``: Texto del fragmento
   * ``metadata``: Información del paper (source, author, year, topic)
   * ``distance``: Distancia L2
   * ``similarity``: Similitud coseno (1 - distance)

**Ejemplo:**

.. code-block:: python

   # Buscar información sobre CD
   results = rag.query_papers(
       "Explica el algoritmo Contrastive Divergence",
       n_results=5
   )
   
   # Analizar resultados
   for i, result in enumerate(results):
       print(f"\nResultado {i+1}:")
       print(f"  Similitud: {result['similarity']:.2%}")
       print(f"  Fuente: {result['metadata']['source']}")
       print(f"  Tema: {result['metadata']['topic']}")
       print(f"  Contenido: {result['content'][:150]}...")

Métodos de Generación
---------------------

generate_response
^^^^^^^^^^^^^^^^^

.. automethod:: src.educational_rag.EducationalRAG.generate_response
   :noindex:

Genera respuesta educativa usando Groq AI.

**Parameters:**
   * ``question`` (str): Pregunta del usuario
   * ``context`` (List[Dict]): Fragmentos relevantes

**Returns:**
   Respuesta generada

**Configuración del prompt:**

.. code-block:: python

   system_prompt = """Eres un asistente educativo experto en 
   Máquinas de Boltzmann Restringidas (RBMs) y Deep Learning. 
   Tu objetivo es explicar conceptos de manera clara y pedagógica, 
   usando los papers científicos como referencia."""
   
   user_prompt = f"""Basándote en los siguientes fragmentos de 
   papers científicos, responde la pregunta:
   
   CONTEXTO:
   {context_text}
   
   PREGUNTA: {question}
   
   Por favor proporciona una respuesta clara, detallada y educativa."""

**Ejemplo:**

.. code-block:: python

   # Pregunta compleja
   question = """¿Cuál es la diferencia entre Contrastive Divergence 
   y Persistent Contrastive Divergence?"""
   
   # Buscar contexto
   context = rag.query_papers(question, n_results=4)
   
   # Generar respuesta
   response = rag.generate_response(question, context)
   
   print(f"Pregunta: {question}\n")
   print(f"Respuesta:\n{response}")

Funciones de Renderizado
-------------------------

render_educational_rag_module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.educational_rag.render_educational_rag_module

Renderiza la interfaz completa del sistema RAG en Streamlit.

**Componentes de la interfaz:**

1. **Información de papers:**
   
   * Lista de papers disponibles
   * Referencias bibliográficas completas
   * Estadísticas de la base de datos

2. **Gestión de papers:**
   
   * Descargar desde URL
   * Editar referencias
   * Eliminar papers

3. **Preguntas sugeridas:**
   
   * Nivel Básico (3 preguntas)
   * Nivel Intermedio (3 preguntas)
   * Nivel Avanzado (3 preguntas)

4. **Chat interactivo:**
   
   * Campo de texto para preguntas
   * Configuración de número de fragmentos
   * Visualización de fragmentos encontrados
   * Respuesta generada con IA

5. **Historial:**
   
   * Últimas 5 conversaciones
   * Preguntas y respuestas guardadas

Papers Científicos Incluidos
-----------------------------

La base de conocimiento incluye información de:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Paper
     - Autor
     - Año
   * - A Practical Guide to Training RBMs
     - Geoffrey E. Hinton
     - 2010
   * - Training Products of Experts by Minimizing CD
     - Geoffrey E. Hinton
     - 2002
   * - Reducing Dimensionality with Neural Networks
     - Hinton & Salakhutdinov
     - 2006
   * - RBMs for Collaborative Filtering
     - Salakhutdinov et al.
     - 2007
   * - Deep Learning of Representations
     - Yoshua Bengio
     - 2013
   * - A Fast Learning Algorithm for DBNs
     - Hinton et al.
     - 2006
   * - Training RBMs using PCD
     - Tijmen Tieleman
     - 2008

Ejemplo de Sesión Educativa
----------------------------

.. code-block:: python

   from src.educational_rag import EducationalRAG
   
   # Inicializar
   rag = EducationalRAG()
   rag.load_papers_to_db()
   
   # Sesión de aprendizaje progresivo
   learning_path = [
       # Nivel 1: Fundamentos
       "¿Qué es una Máquina de Boltzmann Restringida?",
       "¿Cuál es la arquitectura de una RBM?",
       "¿Qué es la función de energía?",
       
       # Nivel 2: Entrenamiento
       "¿Cómo se entrena una RBM?",
       "¿Qué es Contrastive Divergence?",
       "¿Cuáles son los hiperparámetros importantes?",
       
       # Nivel 3: Aplicaciones
       "¿Para qué se usan las RBMs?",
       "¿Qué son las Deep Belief Networks?",
       "¿Cuáles son las limitaciones de las RBMs?"
   ]
   
   # Procesar cada pregunta
   for i, question in enumerate(learning_path, 1):
       print(f"\n{'='*60}")
       print(f"Pregunta {i}/{len(learning_path)}")
       print(f"{'='*60}")
       print(f"❓ {question}\n")
       
       # Buscar y generar respuesta
       context = rag.query_papers(question, n_results=3)
       response = rag.generate_response(question, context)
       
       print(f"💡 {response}\n")
       
       # Mostrar fuentes
       print("📚 Fuentes consultadas:")
       for doc in context:
           meta = doc['metadata']
           print(f"  - {meta['author']} ({meta['year']}): {meta['source']}")

Configuración Avanzada
----------------------

Personalizar Modelo de IA
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Modificar configuración de Groq
   response = rag.groq_client.chat.completions.create(
       model="llama-3.3-70b-versatile",  # O "mixtral-8x7b-32768"
       messages=[...],
       temperature=0.5,  # Más determinístico
       max_tokens=3000,  # Respuestas más largas
       top_p=0.9,
       frequency_penalty=0.0,
       presence_penalty=0.0
   )

Personalizar Búsqueda
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Búsqueda con filtros de metadata
   results = rag.collection.query(
       query_texts=["pregunta"],
       n_results=5,
       where={"author": "Geoffrey E. Hinton"},  # Filtrar por autor
       where_document={"$contains": "energy"}   # Filtrar por contenido
   )

Dependencias Requeridas
-----------------------

.. code-block:: bash

   pip install chromadb groq

**ChromaDB:**
   * Base de datos vectorial
   * Búsqueda por similitud
   * Persistencia local

**Groq:**
   * API de LLMs rápidos
   * Llama 3.3 70B
   * Gratis con límites generosos

Troubleshooting
---------------

Error: "ChromaDB no está instalado"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install chromadb

Error: "Groq no está instalado"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install groq

Error: "No se encontró GROQ_API_KEY"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Crear archivo ``.streamlit/secrets.toml``:

.. code-block:: toml

   GROQ_API_KEY = "gsk_tu_api_key_aqui"

Obtener API key gratis: https://console.groq.com

Ver también
-----------

* :doc:`rbm_model` - Implementación de RBM
* Papers científicos en ``articles/``
* `Groq Console <https://console.groq.com>`_
* `ChromaDB Documentation <https://docs.trychroma.com>`_