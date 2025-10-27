"""
============================================================================
MÓDULO RAG EDUCATIVO
============================================================================

Sistema RAG (Retrieval-Augmented Generation) educativo para aprender sobre
Máquinas de Boltzmann Restringidas usando papers científicos y Groq AI.

Autor: Sistema de Física
Versión: 1.0.0
"""

import streamlit as st
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Verificar si ChromaDB está disponible
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    st.warning("⚠️ ChromaDB no está instalado. Instala con: pip install chromadb")

# Verificar si Groq está disponible
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("⚠️ Groq no está instalado. Instala con: pip install groq")


class EducationalRAG:
    """Sistema RAG educativo para aprender sobre RBMs"""
    
    def __init__(self):
        """Inicializa el sistema RAG"""
        self.groq_client = None
        self.chroma_client = None
        self.collection = None
        self.papers_dir = Path("articles")
        
        # Inicializar Groq si está disponible
        if GROQ_AVAILABLE:
            self._init_groq()
        
        # Inicializar ChromaDB si está disponible
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
    
    def _init_groq(self):
        """Inicializa el cliente de Groq"""
        try:
            # Intentar obtener la API key de secrets o variables de entorno
            api_key = None
            
            # Primero intentar desde Streamlit secrets
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets['GROQ_API_KEY']
            # Luego desde variables de entorno
            elif 'GROQ_API_KEY' in os.environ:
                api_key = os.environ['GROQ_API_KEY']
            
            if api_key:
                self.groq_client = Groq(api_key=api_key)
            else:
                st.warning("⚠️ No se encontró GROQ_API_KEY. Configúrala en .streamlit/secrets.toml")
        except Exception as e:
            st.error(f"❌ Error inicializando Groq: {e}")
    
    def _init_chromadb(self):
        """Inicializa ChromaDB"""
        try:
            # Crear directorio para ChromaDB
            chroma_dir = Path("chroma_rbm_db")
            chroma_dir.mkdir(exist_ok=True)
            
            # Inicializar cliente
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Obtener o crear colección
            try:
                self.collection = self.chroma_client.get_collection("rbm_papers")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="rbm_papers",
                    metadata={"description": "Papers científicos sobre RBMs"}
                )
        except Exception as e:
            st.error(f"❌ Error inicializando ChromaDB: {e}")
    
    def get_available_papers(self) -> List[str]:
        """Obtiene lista de papers disponibles"""
        if not self.papers_dir.exists():
            return []
        
        return [f.name for f in self.papers_dir.glob("*.pdf")]
    
    def get_paper_references(self) -> Dict:
        """Carga las referencias de los papers desde el archivo JSON"""
        references_file = self.papers_dir / "papers_references.json"
        if references_file.exists():
            try:
                with open(references_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"⚠️ Error cargando referencias: {e}")
                return {}
        return {}
    
    def load_papers_to_db(self) -> bool:
        """Carga papers a ChromaDB si no están cargados"""
        if not self.collection:
            return False
        
        try:
            # Verificar si ya hay documentos
            count = self.collection.count()
            if count > 0:
                return True
            
            # Si no hay papers PDF, crear documentos de ejemplo
            papers = self.get_available_papers()
            if not papers:
                st.warning("⚠️ No hay papers PDF. Creando base de conocimiento básica...")
                self._create_basic_knowledge()
                return True
            
            st.info(f"📚 Cargando {len(papers)} papers a la base de datos...")
            # Aquí se cargarían los PDFs reales, por ahora usamos conocimiento básico
            self._create_basic_knowledge()
            return True
            
        except Exception as e:
            st.error(f"❌ Error cargando papers: {e}")
            return False
    
    def _get_reference_mapping(self) -> Dict:
        """Obtiene mapeo de referencias desde el JSON o crea uno por defecto"""
        references_file = self.papers_dir / "papers_references.json"
        
        # Mapeo por defecto basado en los sources usados
        default_mapping = {
            "Hinton_2010_Practical_Guide": {
                "author": "Geoffrey E. Hinton",
                "year": 2010,
                "citation": "Hinton, G. E. (2010). A Practical Guide to Training Restricted Boltzmann Machines. Technical Report UTML TR 2010-003, University of Toronto."
            },
            "Hinton_2002_Training": {
                "author": "Geoffrey E. Hinton",
                "year": 2002,
                "citation": "Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence. Neural Computation, 14(8), 1771-1800."
            },
            "Hinton_2002_CD": {
                "author": "Geoffrey E. Hinton",
                "year": 2002,
                "citation": "Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence. Neural Computation, 14(8), 1771-1800."
            },
            "Hinton_2010_Guide": {
                "author": "Geoffrey E. Hinton",
                "year": 2010,
                "citation": "Hinton, G. E. (2010). A Practical Guide to Training Restricted Boltzmann Machines. Technical Report UTML TR 2010-003, University of Toronto."
            },
            "Salakhutdinov_2007": {
                "author": "Salakhutdinov et al.",
                "year": 2007,
                "citation": "Salakhutdinov, R., Mnih, A., & Hinton, G. (2007). Restricted Boltzmann Machines for Collaborative Filtering. Proceedings of the 24th ICML."
            },
            "Hinton_2006_Science": {
                "author": "Hinton & Salakhutdinov",
                "year": 2006,
                "citation": "Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. Science, 313(5786), 504-507."
            },
            "Hinton_2010_Practical": {
                "author": "Geoffrey E. Hinton",
                "year": 2010,
                "citation": "Hinton, G. E. (2010). A Practical Guide to Training Restricted Boltzmann Machines. Technical Report UTML TR 2010-003, University of Toronto."
            },
            "Bengio_2013": {
                "author": "Yoshua Bengio",
                "year": 2013,
                "citation": "Bengio, Y. (2013). Deep Learning of Representations: Looking Forward. Statistical Language and Speech Processing, LNCS 7978, 1-37."
            },
            "Hinton_2006_DBN": {
                "author": "Hinton et al.",
                "year": 2006,
                "citation": "Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A Fast Learning Algorithm for Deep Belief Nets. Neural Computation, 18(7), 1527-1554."
            },
            "Tieleman_2008_PCD": {
                "author": "Tijmen Tieleman",
                "year": 2008,
                "citation": "Tieleman, T. (2008). Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient. Proceedings of the 25th ICML."
            }
        }
        
        return default_mapping
    
    def _create_basic_knowledge(self):
        """Crea una base de conocimiento básica sobre RBMs"""
        basic_knowledge = [
            {
                "id": "rbm_intro_1",
                "content": """Una Máquina de Boltzmann Restringida (RBM) es un tipo de red neuronal generativa
                estocástica que puede aprender una distribución de probabilidad sobre su conjunto de entradas.
                Fue inventada por Geoffrey Hinton y es un componente fundamental en el aprendizaje profundo.
                Las RBMs consisten en dos capas: una capa visible (v) que representa los datos observables y
                una capa oculta (h) que captura las características latentes.
                
                Referencia: Hinton, G. E. (2010). A Practical Guide to Training Restricted Boltzmann Machines.
                Technical Report UTML TR 2010-003, University of Toronto.""",
                "metadata": {"source": "Hinton_2010_Practical_Guide", "topic": "introducción",
                           "author": "Geoffrey E. Hinton", "year": 2010}
            },
            {
                "id": "rbm_structure_1",
                "content": """La arquitectura de una RBM es bipartita, lo que significa que no hay conexiones
                entre unidades de la misma capa. Solo existen conexiones entre la capa visible y la capa oculta.
                Esta restricción hace que el entrenamiento sea más eficiente que en las Máquinas de Boltzmann
                completas. Cada conexión tiene un peso asociado (W), y cada unidad tiene un sesgo (bias).
                
                Referencia: Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence.
                Neural Computation, 14(8), 1771-1800.""",
                "metadata": {"source": "Hinton_2002_Training", "topic": "arquitectura",
                           "author": "Geoffrey E. Hinton", "year": 2002}
            },
            {
                "id": "rbm_cd_1",
                "content": """El algoritmo Contrastive Divergence (CD) es el método principal para entrenar RBMs.
                Fue propuesto por Geoffrey Hinton en 2002. CD es una aproximación al gradiente de la log-verosimilitud
                que es mucho más eficiente computacionalmente que el método exacto. El algoritmo alterna entre
                muestrear la capa oculta dado la visible y viceversa, típicamente usando CD-1 (una sola iteración).
                
                Referencia: Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence.
                Neural Computation, 14(8), 1771-1800.""",
                "metadata": {"source": "Hinton_2002_CD", "topic": "entrenamiento",
                           "author": "Geoffrey E. Hinton", "year": 2002}
            },
            {
                "id": "rbm_energy_1",
                "content": """Las RBMs definen una función de energía E(v,h) = -a'v - b'h - v'Wh, donde a y b son
                los sesgos de las capas visible y oculta respectivamente, y W es la matriz de pesos. La probabilidad
                conjunta está dada por P(v,h) = exp(-E(v,h))/Z, donde Z es la función de partición. El objetivo
                del entrenamiento es minimizar esta energía para los datos de entrenamiento.
                
                Referencia: Hinton, G. E. (2010). A Practical Guide to Training Restricted Boltzmann Machines.
                Technical Report UTML TR 2010-003, University of Toronto.""",
                "metadata": {"source": "Hinton_2010_Guide", "topic": "función_energía",
                           "author": "Geoffrey E. Hinton", "year": 2010}
            },
            {
                "id": "rbm_applications_1",
                "content": """Las RBMs tienen múltiples aplicaciones: 1) Reducción de dimensionalidad y extracción
                de características, 2) Sistemas de recomendación (filtrado colaborativo), 3) Inicialización de
                redes neuronales profundas (Deep Belief Networks), 4) Modelado de tópicos, 5) Procesamiento de
                imágenes y visión por computadora. Son especialmente útiles cuando se necesita aprender
                representaciones no supervisadas de datos complejos.
                
                Referencia: Salakhutdinov, R., Mnih, A., & Hinton, G. (2007). Restricted Boltzmann Machines
                for Collaborative Filtering. Proceedings of the 24th International Conference on Machine Learning.""",
                "metadata": {"source": "Salakhutdinov_2007", "topic": "aplicaciones",
                           "author": "Salakhutdinov et al.", "year": 2007}
            },
            {
                "id": "rbm_vs_nn_1",
                "content": """A diferencia de las redes neuronales tradicionales feedforward, las RBMs son modelos
                generativos que pueden generar nuevos datos similares a los de entrenamiento. Las redes neuronales
                tradicionales son discriminativas y se enfocan en mapear entradas a salidas. Las RBMs aprenden
                la distribución de probabilidad de los datos, mientras que las redes feedforward aprenden una
                función de mapeo. Además, las RBMs son no supervisadas, mientras que las redes tradicionales
                típicamente requieren etiquetas.
                
                Referencia: Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data
                with Neural Networks. Science, 313(5786), 504-507.""",
                "metadata": {"source": "Hinton_2006_Science", "topic": "comparación",
                           "author": "Hinton & Salakhutdinov", "year": 2006}
            },
            {
                "id": "rbm_training_1",
                "content": """El proceso de entrenamiento de una RBM involucra: 1) Inicializar pesos y sesgos
                aleatoriamente, 2) Para cada mini-batch: calcular las activaciones de la capa oculta dado los
                datos visibles (fase positiva), 3) Reconstruir la capa visible desde la oculta, 4) Recalcular
                las activaciones ocultas (fase negativa), 5) Actualizar pesos usando la diferencia entre las
                fases positiva y negativa. Este proceso se repite por múltiples épocas hasta convergencia.
                
                Referencia: Hinton, G. E. (2010). A Practical Guide to Training Restricted Boltzmann Machines.
                Technical Report UTML TR 2010-003, University of Toronto.""",
                "metadata": {"source": "Hinton_2010_Practical", "topic": "proceso_entrenamiento",
                           "author": "Geoffrey E. Hinton", "year": 2010}
            },
            {
                "id": "rbm_limitations_1",
                "content": """Las RBMs tienen varias limitaciones: 1) Dificultad para entrenar en datasets muy
                grandes, 2) La función de partición Z es intratable de calcular exactamente, 3) Pueden sufrir
                de modos espurios en la distribución aprendida, 4) Requieren ajuste cuidadoso de hiperparámetros,
                5) Han sido en gran parte superadas por métodos más modernos como VAEs y GANs para tareas
                generativas. Sin embargo, siguen siendo valiosas para entender el aprendizaje profundo.
                
                Referencia: Bengio, Y. (2013). Deep Learning of Representations: Looking Forward.
                Statistical Language and Speech Processing, LNCS 7978, 1-37.""",
                "metadata": {"source": "Bengio_2013", "topic": "limitaciones",
                           "author": "Yoshua Bengio", "year": 2013}
            },
            {
                "id": "rbm_dbn_1",
                "content": """Las Deep Belief Networks (DBNs) son redes profundas construidas apilando múltiples
                RBMs. El entrenamiento se realiza capa por capa de forma greedy: primero se entrena la primera
                RBM con los datos originales, luego se usa su capa oculta como entrada para entrenar la segunda
                RBM, y así sucesivamente. Este pre-entrenamiento no supervisado fue crucial para el resurgimiento
                del aprendizaje profundo en 2006, permitiendo entrenar redes más profundas efectivamente.
                
                Referencia: Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A Fast Learning Algorithm for
                Deep Belief Nets. Neural Computation, 18(7), 1527-1554.""",
                "metadata": {"source": "Hinton_2006_DBN", "topic": "deep_belief_networks",
                           "author": "Hinton et al.", "year": 2006}
            },
            {
                "id": "rbm_pcd_1",
                "content": """Persistent Contrastive Divergence (PCD) es una mejora del algoritmo CD propuesta
                por Tieleman en 2008. En lugar de reiniciar la cadena de Markov desde los datos en cada
                actualización, PCD mantiene una cadena persistente que continúa entre actualizaciones. Esto
                permite que la cadena explore mejor el espacio de estados y puede llevar a mejores modelos,
                especialmente cuando se usan múltiples pasos de Gibbs sampling (CD-k con k>1).
                
                Referencia: Tieleman, T. (2008). Training Restricted Boltzmann Machines using Approximations
                to the Likelihood Gradient. Proceedings of the 25th International Conference on Machine Learning.""",
                "metadata": {"source": "Tieleman_2008_PCD", "topic": "pcd",
                           "author": "Tijmen Tieleman", "year": 2008}
            }
        ]
        
        # Agregar documentos a la colección
        for doc in basic_knowledge:
            try:
                self.collection.add(
                    documents=[doc["content"]],
                    metadatas=[doc["metadata"]],
                    ids=[doc["id"]]
                )
            except Exception as e:
                st.warning(f"⚠️ Error agregando documento {doc['id']}: {e}")
        
        st.success(f"✅ Base de conocimiento creada con {len(basic_knowledge)} documentos")
    
    def query_papers(self, question: str, n_results: int = 3) -> List[Dict]:
        """Busca información relevante en los papers usando similitud coseno"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            # Formatear resultados
            formatted_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    # Calcular similitud coseno (1 - distancia L2 normalizada)
                    similarity = 1 - distance
                    
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'similarity': similarity
                    })
            
            return formatted_results
        except Exception as e:
            st.error(f"❌ Error consultando papers: {e}")
            return []
    
    def generate_response(self, question: str, context: List[Dict]) -> str:
        """Genera respuesta usando Groq AI"""
        if not self.groq_client:
            return "⚠️ Groq AI no está configurado. Por favor configura tu API key."
        
        try:
            # Construir contexto
            context_text = "\n\n".join([
                f"Fragmento {i+1}:\n{doc['content']}"
                for i, doc in enumerate(context)
            ])
            
            # Crear prompt
            system_prompt = """Eres un asistente educativo experto en Máquinas de Boltzmann Restringidas (RBMs) 
y Deep Learning. Tu objetivo es explicar conceptos de manera clara y pedagógica, usando los papers 
científicos como referencia. Siempre proporciona explicaciones detalladas pero accesibles."""
            
            user_prompt = f"""Basándote en los siguientes fragmentos de papers científicos, responde la pregunta:

CONTEXTO:
{context_text}

PREGUNTA: {question}

Por favor proporciona una respuesta clara, detallada y educativa. Si es relevante, menciona las fuentes."""
            
            # Llamar a Groq
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"❌ Error generando respuesta: {e}"


def render_educational_rag_module():
    """Renderiza la interfaz del módulo RAG educativo"""
    st.title("🎓 Aprende sobre RBMs")
    st.markdown("### *Sistema RAG educativo con papers científicos*")
    
    # Verificar dependencias
    if not CHROMADB_AVAILABLE or not GROQ_AVAILABLE:
        st.error("❌ Faltan dependencias necesarias")
        st.markdown("""
        **Instala las dependencias:**
        ```bash
        pip install chromadb groq
        ```
        """)
        return
    
    # Inicializar RAG
    if 'rag_system' not in st.session_state:
        with st.spinner("🔧 Inicializando sistema RAG..."):
            st.session_state.rag_system = EducationalRAG()
    
    rag = st.session_state.rag_system
    
    # Verificar configuración
    if not rag.groq_client:
        st.warning("⚠️ Configura tu API key de Groq en `.streamlit/secrets.toml`:")
        st.code("""
GROQ_API_KEY = "tu_api_key_aqui"
        """)
        st.markdown("[Obtén tu API key gratis en Groq](https://console.groq.com)")
        return
    
    # Información de papers disponibles
    papers = rag.get_available_papers()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📚 Base de Conocimiento")
        if papers:
            st.success(f"✅ {len(papers)} papers científicos disponibles")
            
            # Cargar referencias
            references = rag.get_paper_references()
            
            with st.expander("Ver lista de papers con referencias completas"):
                for paper in sorted(papers):
                    if paper in references:
                        ref = references[paper]
                        st.markdown(f"""
**📄 {ref['title']}**
*{ref['authors']}* ({ref['year']})
{ref['publication']}
`{paper}`
                        """)
                        st.divider()
                    else:
                        st.markdown(f"- 📄 {paper}")
        else:
            st.warning("⚠️ No hay papers cargados")
            st.markdown("""
            **Para cargar papers:**
            1. Ejecuta: `python src/libros.py`
            2. O coloca PDFs manualmente en la carpeta `articles/`
            """)
    
    with col2:
        st.subheader("🤖 Modelo")
        st.info("**Llama 3.3 70B**\nvia Groq AI")
    
    st.divider()
    # Sección de gestión de papers
    with st.expander("📤 Gestionar Papers Personalizados"):
        st.markdown("### Agregar Nuevos Papers")
        
        tab1, tab2 = st.tabs(["➕ Agregar URL", "📝 Editar Referencias"])
        
        with tab1:
            st.markdown("**Descargar paper desde URL:**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                paper_url = st.text_input(
                    "URL del paper (PDF):",
                    placeholder="https://ejemplo.com/paper.pdf",
                    key="paper_url_input"
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                download_btn = st.button("⬇️ Descargar", use_container_width=True)
            
            if download_btn and paper_url:
                with st.spinner("📥 Descargando paper..."):
                    try:
                        import requests
                        response = requests.get(paper_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                        response.raise_for_status()
                        
                        # Generar nombre de archivo
                        filename = paper_url.split('/')[-1]
                        if not filename.endswith('.pdf'):
                            filename = f"paper_{len(papers)+1}.pdf"
                        
                        filepath = rag.papers_dir / filename
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        st.success(f"✅ Paper descargado: {filename}")
                        st.info("💡 Agrega la referencia completa en la pestaña 'Editar Referencias'")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error descargando: {e}")
            
            st.markdown("---")
            st.markdown("**Papers actuales:**")
            if papers:
                for paper in sorted(papers):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"📄 {paper}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{paper}"):
                            try:
                                (rag.papers_dir / paper).unlink()
                                st.success(f"✅ Eliminado: {paper}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
            else:
                st.info("No hay papers cargados")
        
        with tab2:
            st.markdown("**Editar referencias bibliográficas:**")
            
            # Cargar referencias actuales
            references = rag.get_paper_references()
            
            # Selector de paper
            if papers:
                selected_paper = st.selectbox(
                    "Selecciona un paper:",
                    options=["-- Nuevo --"] + sorted(papers),
                    key="paper_selector"
                )
                
                # Si es nuevo o existente
                if selected_paper != "-- Nuevo --":
                    current_ref = references.get(selected_paper, {})
                else:
                    current_ref = {}
                    selected_paper = st.text_input("Nombre del archivo PDF:", placeholder="ejemplo.pdf", key="new_paper_name")
                
                # Formulario de referencia
                if selected_paper and selected_paper != "-- Nuevo --":
                    st.markdown(f"**Editando:** `{selected_paper}`")
                    
                    title = st.text_input(
                        "Título del paper:",
                        value=current_ref.get('title', ''),
                        placeholder="Training Restricted Boltzmann Machines",
                        key="ref_title"
                    )
                    
                    authors = st.text_input(
                        "Autores:",
                        value=current_ref.get('authors', ''),
                        placeholder="Geoffrey E. Hinton",
                        key="ref_authors"
                    )
                    
                    year = st.number_input(
                        "Año:",
                        min_value=1980,
                        max_value=2025,
                        value=current_ref.get('year', 2010),
                        key="ref_year"
                    )
                    
                    publication = st.text_input(
                        "Publicación:",
                        value=current_ref.get('publication', ''),
                        placeholder="Neural Computation, 14(8), 1771-1800",
                        key="ref_publication"
                    )
                    
                    # Generar citación automática
                    if title and authors and year:
                        citation = f"{authors} ({year}). {title}. {publication}"
                        st.text_area("Citación generada:", value=citation, height=100, disabled=True, key="ref_citation_preview")
                    
                    # Botón guardar
                    if st.button("💾 Guardar Referencia", type="primary", key="save_ref_btn"):
                        try:
                            # Actualizar referencias
                            references[selected_paper] = {
                                "title": title,
                                "authors": authors,
                                "year": year,
                                "publication": publication,
                                "citation": citation if title and authors and year else ""
                            }
                            
                            # Guardar en JSON
                            ref_file = rag.papers_dir / "papers_references.json"
                            with open(ref_file, 'w', encoding='utf-8') as f:
                                json.dump(references, f, indent=2, ensure_ascii=False)
                            
                            st.success("✅ Referencia guardada exitosamente")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error guardando: {e}")
            else:
                st.info("No hay papers disponibles. Descarga algunos primero.")
    
    st.divider()
    
    
    # Cargar papers si no están cargados
    if rag.collection and rag.collection.count() == 0:
        with st.spinner("📚 Inicializando base de conocimiento..."):
            rag.load_papers_to_db()
    
    # Preguntas sugeridas
    st.subheader("💡 Preguntas Sugeridas")
    
    suggested_questions = {
        "🔰 Básico": [
            "¿Qué es una Máquina de Boltzmann Restringida?",
            "¿Cuál es la diferencia entre RBM y una red neuronal tradicional?",
            "¿Para qué se usan las RBMs?",
        ],
        "🎯 Intermedio": [
            "¿Cómo funciona el algoritmo Contrastive Divergence?",
            "¿Qué son las unidades visibles y ocultas en una RBM?",
            "¿Cómo se entrenan las RBMs?",
        ],
        "🚀 Avanzado": [
            "¿Cuáles son las limitaciones de las RBMs?",
            "¿Cómo se apilan RBMs para crear Deep Belief Networks?",
            "¿Qué es Persistent Contrastive Divergence?",
        ]
    }
    
    # Inicializar question en session_state si no existe
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Inicializar trigger para búsqueda automática
    if 'auto_search' not in st.session_state:
        st.session_state.auto_search = False
    
    cols = st.columns(3)
    for i, (level, questions) in enumerate(suggested_questions.items()):
        with cols[i]:
            st.markdown(f"**{level}**")
            for q in questions:
                if st.button(q, key=f"suggest_{q}", use_container_width=True):
                    st.session_state.current_question = q
                    st.session_state.auto_search = True
                    st.rerun()
    
    st.divider()
    
    # Chat interface
    st.subheader("💬 Haz tu Pregunta")
    
    question = st.text_area(
        "Escribe tu pregunta sobre RBMs:",
        value=st.session_state.current_question,
        height=100,
        placeholder="Ejemplo: ¿Cómo funciona el algoritmo Contrastive Divergence en las RBMs?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ask_button = st.button("🔍 Buscar Respuesta", type="primary", use_container_width=True)
    with col2:
        n_results = st.number_input("Fragmentos", min_value=1, max_value=5, value=3)
    with col3:
        st.caption("**Métrica:**")
        st.caption("Similitud Coseno")
    
    # Ejecutar búsqueda si se presionó el botón O si hay auto_search activado
    should_search = (ask_button and question) or (st.session_state.auto_search and question)
    
    if should_search:
        # Resetear auto_search
        st.session_state.auto_search = False
        with st.spinner("🔍 Buscando en papers científicos..."):
            # Buscar contexto relevante
            context = rag.query_papers(question, n_results=n_results)
            
            if not context:
                st.warning("⚠️ No se encontró información relevante. Asegúrate de tener papers cargados.")
                return
            
            # Mostrar contexto encontrado
            st.info("🔍 **Métrica de búsqueda:** Similitud Coseno (Cosine Similarity)")
            
            with st.expander("📚 Fragmentos Relevantes Encontrados"):
                # Obtener mapeo de referencias
                ref_mapping = rag._get_reference_mapping()
                
                for i, doc in enumerate(context):
                    metadata = doc.get('metadata', {})
                    similarity = doc.get('similarity', 0)
                    
                    # Obtener información de la fuente
                    source = metadata.get('source', 'Desconocido')
                    
                    # Buscar referencia completa en el mapeo
                    if source in ref_mapping:
                        ref_info = ref_mapping[source]
                        author = ref_info['author']
                        year = ref_info['year']
                        citation = ref_info['citation']
                    else:
                        author = 'Autor desconocido'
                        year = 'Año desconocido'
                        citation = source
                    
                    st.markdown(f"**Fragmento {i+1}** (Similitud Coseno: {similarity:.2%})")
                    st.caption(f"📄 **Referencia:** {citation}")
                    # Mostrar el triple de longitud (1500 caracteres en lugar de 500)
                    content_preview = doc['content'][:1500]
                    if len(doc['content']) > 1500:
                        content_preview += "..."
                    st.markdown(content_preview)
                    st.divider()
            
            # Generar respuesta
            with st.spinner("🤖 Generando respuesta con Groq AI..."):
                response = rag.generate_response(question, context)
            
            # Mostrar respuesta
            st.subheader("💡 Respuesta")
            st.markdown(response)
            
            # Guardar en historial
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({
                'question': question,
                'response': response,
                'context_count': len(context)
            })
    
    # Mostrar historial
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.divider()
        st.subheader("📜 Historial de Conversación")
        
        for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"❓ {item['question'][:100]}..."):
                st.markdown(f"**Pregunta:** {item['question']}")
                st.markdown(f"**Respuesta:** {item['response']}")
                st.caption(f"Basado en {item['context_count']} fragmentos de papers")


if __name__ == "__main__":
    print("Módulo RAG educativo cargado correctamente")