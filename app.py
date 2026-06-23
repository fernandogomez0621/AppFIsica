"""
============================================================================
APLICACIÓN PRINCIPAL - Sistema de Riesgo Crediticio con RBM y RAG Educativo
============================================================================

Aplicación completa de análisis y predicción de riesgo crediticio hipotecario 
para Colombia usando Streamlit, con:

1. Simulación y análisis de datos crediticios
2. Máquina de Boltzmann Restringida (RBM) como extractor de características
3. Modelos de clasificación supervisada
4. Sistema RAG educativo con Groq para aprender sobre RBMs

Autor: Sistema de Física
Versión: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys
import os
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

# Importar módulos locales
try:
    from src.generar_datos import GeneradorCreditoHipotecarioRealista, generar_datos_credito_realista
except ImportError:
    st.error("❌ Error importando el generador de datos. Verifica que generar_datos.py esté en src/.")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="🏦 Sistema de Riesgo Crediticio con RBM",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tu-repo/help',
        'Report a bug': 'https://github.com/tu-repo/issues',
        'About': """
        # Sistema de Riesgo Crediticio con RBM
        
        Aplicación completa para análisis de riesgo crediticio hipotecario
        con Máquinas de Boltzmann Restringidas y sistema RAG educativo.
        
        **Autor:**
        Andrés Fernando Gómez Rojas
        Pregrado en Física
        Universidad Distrital Francisco José de Caldas
        
        **Director:**
        Carlos Andrés Gómez Vasco
        
        **Características:**
        - Generación de datos sintéticos realistas
        - Análisis exploratorio avanzado
        - Máquinas de Boltzmann Restringidas (RBM)
        - Modelos de Machine Learning supervisados
        - Sistema RAG educativo con papers científicos
        
        **Versión:** 1.0.0
        """
    }
)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_data
def load_sample_data():
    """Carga datos de muestra para demostración"""
    try:
        # Intentar cargar datos existentes
        if os.path.exists("data/processed/datos_credito_hipotecario_realista.csv"):
            return pd.read_csv("data/processed/datos_credito_hipotecario_realista.csv")
        else:
            return None
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

def show_app_info():
    """Muestra información general de la aplicación"""
    st.markdown("""
    ## 🎯 Objetivo del Proyecto
    
    Crear un sistema integral que permita:
    
    - 📊 **Generar/cargar datos** de solicitudes de crédito hipotecario
    - 📈 **Realizar análisis exploratorio** avanzado
    - ⚙️ **Aplicar ingeniería de características** automática
    - 🧠 **Entrenar modelos predictivos** con RBM + clasificadores
    - 🔮 **Predecir riesgo crediticio** en nuevos solicitantes
    - 🎓 **Aprender sobre Máquinas de Boltzmann** mediante un asistente RAG
    
    ## 📋 Variables del Sistema
    
    ### Variables Financieras del Crédito:
    - `valor_inmueble`: Valor comercial de la propiedad (COP)
    - `monto_credito`: Monto solicitado del préstamo (COP)
    - `cuota_inicial`: Porcentaje de cuota inicial (%)
    - `plazo_credito`: Plazo del crédito en años
    - `tasa_interes`: Tasa de interés anual (%)
    
    ### Perfil Financiero del Solicitante:
    - `puntaje_datacredito`: Score crediticio (150-950)
    - `salario_mensual`: Ingreso mensual (COP)
    - `egresos_mensuales`: Gastos mensuales totales (COP)
    - `saldo_promedio_banco`: Saldo promedio últimos 6 meses (COP)
    - `patrimonio_total`: Patrimonio neto (COP)
    - `numero_propiedades`: Cantidad de propiedades que posee
    - `numero_demandas`: Demandas legales por dinero
    
    ### Variable Objetivo:
    - `nivel_riesgo`: **Bajo** / **Medio** / **Alto**
    """)

def render_home():
    """Renderiza la página de inicio"""
    st.title("🏦 Sistema de Riesgo Crediticio con RBM")
    st.markdown("### *Análisis Predictivo + Máquinas de Boltzmann + RAG Educativo*")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Precisión Objetivo",
            value="95%+",
            help="Precisión esperada del modelo RBM + Clasificador"
        )
    
    with col2:
        st.metric(
            label="📊 Variables",
            value="50+",
            help="Variables originales + características ingenierizadas"
        )
    
    with col3:
        st.metric(
            label="🧠 Modelos",
            value="9",
            help="Diferentes algoritmos de ML implementados"
        )
    
    with col4:
        st.metric(
            label="📚 Papers",
            value="15+",
            help="Papers científicos en el sistema RAG"
        )
    
    st.divider()
    
    # Información de la aplicación
    show_app_info()
    
    # Estado del sistema
    st.divider()
    st.subheader("📊 Estado del Sistema")
    
    # Verificar datos
    sample_data = load_sample_data()
    if sample_data is not None:
        st.success(f"✅ Datos cargados: {len(sample_data):,} registros")
        
        # Mostrar distribución de riesgo
        if 'nivel_riesgo' in sample_data.columns:
            fig = px.pie(
                sample_data, 
                names='nivel_riesgo',
                title="Distribución de Nivel de Riesgo",
                color_discrete_map={
                    'Bajo': '#28a745',
                    'Medio': '#ffc107', 
                    'Alto': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No hay datos cargados. Ve a 'Generar Datos' para crear un dataset.")
    
    # Verificar modelos
    if os.path.exists("models/"):
        model_files = list(Path("models").rglob("*.pkl"))
        if model_files:
            st.success(f"✅ Modelos entrenados: {len(model_files)}")
        else:
            st.info("ℹ️ No hay modelos entrenados aún.")
    
    # Verificar papers
    if os.path.exists("articles/"):
        pdf_files = list(Path("articles").glob("*.pdf"))
        if pdf_files:
            st.success(f"✅ Papers científicos: {len(pdf_files)}")
        else:
            st.info("ℹ️ No hay papers cargados. Ve a 'Aprende sobre RBMs' para agregar papers.")

def render_data_generator():
    """Renderiza el módulo de generación de datos"""
    st.title("📊 Generador de Datos Sintéticos")
    st.markdown("### *Genera datasets realistas de crédito hipotecario para Colombia*")
    
    # Configuración
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Configuración del Dataset")
        
        n_registros = st.slider(
            "Número de registros",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Cantidad de solicitudes de crédito a generar"
        )
        
        semilla = st.number_input(
            "Semilla aleatoria",
            min_value=1,
            max_value=9999,
            value=42,
            help="Para reproducibilidad de resultados"
        )
        
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            exportar_csv = st.checkbox("Exportar CSV", value=True)
        with col_export2:
            exportar_metadata = st.checkbox("Exportar Metadata", value=True)
    
    with col2:
        st.subheader("📈 Distribución Objetivo")
        st.markdown("""
        **Distribución Realista:**
        - 🟢 **Bajo:** 60%
        - 🟡 **Medio:** 25% 
        - 🔴 **Alto:** 15%
        
        **Características:**
        - ✅ Correlaciones realistas
        - ✅ Capacidad residual positiva
        - ✅ DTI máximo 35%
        - ✅ Valores colombianos
        """)
    
    # Botón de generación
    if st.button("🚀 Generar Dataset", type="primary", use_container_width=True):
        with st.spinner("⏳ Generando datos sintéticos..."):
            try:
                # Generar datos
                df = generar_datos_credito_realista(
                    n_registros=n_registros,
                    semilla=semilla,
                    exportar_csv=exportar_csv,
                    exportar_metadata=exportar_metadata
                )
                
                # Guardar en directorio de datos procesados
                os.makedirs("data/processed", exist_ok=True)
                df.to_csv("data/processed/datos_credito_hipotecario_realista.csv", index=False)
                
                st.success(f"✅ Dataset generado exitosamente: {len(df):,} registros")
                
                # Mostrar estadísticas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Registros", f"{len(df):,}")
                
                with col2:
                    st.metric("Variables", len(df.columns))
                
                with col3:
                    riesgo_dist = df['nivel_riesgo'].value_counts(normalize=True) * 100
                    st.metric("Riesgo Bajo", f"{riesgo_dist.get('Bajo', 0):.1f}%")
                
                # Mostrar muestra
                st.subheader("📋 Muestra de Datos Generados")
                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    height=400
                )
                
                # Gráficos de distribución
                st.subheader("📊 Distribuciones Principales")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribución de riesgo
                    fig_riesgo = px.pie(
                        df, 
                        names='nivel_riesgo',
                        title="Distribución de Nivel de Riesgo",
                        color_discrete_map={
                            'Bajo': '#28a745',
                            'Medio': '#ffc107', 
                            'Alto': '#dc3545'
                        }
                    )
                    st.plotly_chart(fig_riesgo, use_container_width=True)
                
                with col2:
                    # Distribución de salarios
                    fig_salario = px.histogram(
                        df,
                        x='salario_mensual',
                        title="Distribución de Salarios",
                        nbins=50,
                        labels={'salario_mensual': 'Salario Mensual (COP)'}
                    )
                    fig_salario.update_layout(showlegend=False)
                    st.plotly_chart(fig_salario, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error generando datos: {e}")
                st.exception(e)

def render_placeholder_module(title, description, features):
    """Renderiza un módulo placeholder"""
    st.title(title)
    st.markdown(f"### *{description}*")
    
    st.info("🚧 **Módulo en construcción**")
    
    st.markdown("**Características planificadas:**")
    for feature in features:
        st.markdown(f"- {feature}")
    
    st.markdown("---")
    st.markdown("💡 **Próximamente:** Este módulo estará disponible en la próxima versión.")

# ============================================================================
# NAVEGACIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación"""
    
    # Sidebar de navegación
    with st.sidebar:
        st.title("🏦 Navegación")
        st.markdown("---")
        
        page = st.radio(
            "Selecciona un módulo:",
            [
                "🏠 Inicio",
                "📊 Generar Datos",
                "📁 Cargar Datos",
                "📈 Análisis Descriptivo",
                "🔧 Ingeniería de Características",
                "🎯 Clustering",
                "⚡ Máquina de Boltzmann (RBM)",
                "🤖 Modelos Supervisados",
                "📊 Comparación de Modelos",
                "🔮 Predicción",
                "🔄 Re-entrenamiento",
                "🎓 Aprende sobre RBMs",
                "📚 Documentación"
            ],
            key="navigation"
        )
        
        st.markdown("---")
        
        # Información del sistema
        st.markdown("### 📊 Estado del Sistema")
        
        # Verificar datos
        if os.path.exists("data/processed/datos_credito_hipotecario_realista.csv"):
            try:
                df = pd.read_csv("data/processed/datos_credito_hipotecario_realista.csv")
                st.success(f"✅ Datos: {len(df):,} registros")
            except:
                st.warning("⚠️ Error cargando datos")
        else:
            st.warning("⚠️ Sin datos")
        
        # Verificar modelos
        model_files = list(Path("models").rglob("*.pkl")) if os.path.exists("models") else []
        if model_files:
            st.success(f"✅ Modelos: {len(model_files)}")
        else:
            st.info("ℹ️ Sin modelos")
        
        # Verificar papers
        pdf_files = list(Path("articles").glob("*.pdf")) if os.path.exists("articles") else []
        if pdf_files:
            st.success(f"✅ Papers: {len(pdf_files)}")
        else:
            st.info("ℹ️ Sin papers")
    
    # Renderizar página seleccionada
    if page == "🏠 Inicio":
        render_home()
    
    elif page == "📊 Generar Datos":
        render_data_generator()
    
    elif page == "📁 Cargar Datos":
        # Importar y renderizar el módulo de procesamiento de datos
        try:
            from src.data_processor import render_data_processor_module
            render_data_processor_module()
        except ImportError:
            st.error("❌ Error importando el módulo de procesamiento de datos")
            render_placeholder_module(
                "📁 Cargar Datos",
                "Carga y validación de datasets externos",
                [
                    "📤 Carga de archivos CSV, Excel, Parquet",
                    "✅ Validación automática de datos",
                    "🔍 Detección de outliers y valores faltantes",
                    "📊 Reporte de calidad de datos",
                    "🔧 Limpieza y preprocesamiento automático"
                ]
            )
    
    elif page == "📈 Análisis Descriptivo":
        # Crear tabs para análisis univariado y bivariado
        analysis_tab1, analysis_tab2 = st.tabs(["📊 Análisis Univariado", "🔗 Análisis Bivariado"])
        
        with analysis_tab1:
            # Importar y renderizar el módulo de análisis univariado
            try:
                from src.univariate_analysis import render_univariate_module
                render_univariate_module()
            except ImportError:
                st.error("❌ Error importando el módulo de análisis univariado")
        
        with analysis_tab2:
            # Importar y renderizar el módulo de análisis bivariado
            try:
                from src.bivariate_analysis import render_bivariate_module
                render_bivariate_module()
            except ImportError:
                st.error("❌ Error importando el módulo de análisis bivariado")
    
    elif page == "🔧 Ingeniería de Características":
        # Importar y renderizar el módulo de ingeniería de características
        try:
            from src.feature_engineering import render_feature_engineering_module
            render_feature_engineering_module()
        except ImportError:
            st.error("❌ Error importando el módulo de ingeniería de características")
            render_placeholder_module(
                "🔧 Ingeniería de Características",
                "Creación automática de variables derivadas",
                [
                    "💰 Ratios financieros (LTV, DTI, etc.)",
                    "📊 Indicadores de riesgo",
                    "🔗 Variables de interacción",
                    "📈 Transformaciones matemáticas",
                    "🎯 Binning y discretización inteligente"
                ]
            )
    
    elif page == "🎯 Clustering":
        # Importar y renderizar el módulo de clustering
        try:
            from src.clustering import render_clustering_module
            render_clustering_module()
        except ImportError:
            st.error("❌ Error importando el módulo de clustering")
            render_placeholder_module(
                "🎯 Clustering",
                "Segmentación de solicitantes en grupos homogéneos",
                [
                    "🔍 Determinación automática de K óptimo",
                    "📊 Múltiples algoritmos (K-Means, Hierarchical, DBSCAN)",
                    "📈 Visualizaciones PCA 2D/3D interactivas",
                    "📋 Perfiles detallados por cluster",
                    "🎯 Etiquetado automático de riesgo"
                ]
            )
    
    elif page == "⚡ Máquina de Boltzmann (RBM)":
        # Importar y renderizar el módulo RBM
        try:
            from src.rbm_model import render_rbm_module
            render_rbm_module()
        except ImportError:
            st.error("❌ Error importando el módulo RBM")
            render_placeholder_module(
                "⚡ Máquina de Boltzmann Restringida",
                "Extracción de características latentes con RBM",
                [
                    "🧠 Implementación completa de RBM desde cero",
                    "⚡ Algoritmo Contrastive Divergence (CD-k)",
                    "📊 Métricas de evaluación (error reconstrucción, pseudo log-likelihood)",
                    "🎨 Visualizaciones de pesos y activaciones",
                    "🔧 Hiperparámetros configurables interactivamente"
                ]
            )
    
    elif page == "🤖 Modelos Supervisados":
        # Importar y renderizar el módulo de modelos supervisados
        try:
            from src.supervised_models import render_supervised_models_module
            render_supervised_models_module()
        except ImportError:
            st.error("❌ Error importando el módulo de modelos supervisados")
            render_placeholder_module(
                "🤖 Modelos Supervisados",
                "Entrenamiento y evaluación de modelos de clasificación",
                [
                    "🎯 6 algoritmos diferentes (Logistic, RF, XGBoost, etc.)",
                    "📊 Evaluación completa con múltiples métricas",
                    "🔧 Optimización automática de hiperparámetros",
                    "💾 Versionado y persistencia de modelos",
                    "🔍 Análisis de importancia de características"
                ]
            )
    
    elif page == "📊 Comparación de Modelos":
        # Usar el mismo módulo de modelos supervisados para comparación
        try:
            from src.supervised_models import render_supervised_models_module
            st.info("💡 La comparación de modelos está integrada en 'Modelos Supervisados'")
            render_supervised_models_module()
        except ImportError:
            render_placeholder_module(
                "📊 Comparación de Modelos",
                "Análisis comparativo de rendimiento de modelos",
                [
                    "📈 Tablas comparativas de métricas",
                    "📊 Gráficos de barras y curvas ROC",
                    "🏆 Ranking automático de modelos",
                    "📋 Tests estadísticos de significancia",
                    "🎯 Selección del mejor modelo para producción"
                ]
            )
    
    elif page == "🔮 Predicción":
        # Importar y renderizar el módulo de predicción
        try:
            from src.prediction import render_prediction_module
            render_prediction_module()
        except ImportError:
            st.error("❌ Error importando el módulo de predicción")
            render_placeholder_module(
                "🔮 Predicción",
                "Sistema de predicción de riesgo crediticio",
                [
                    "📝 Formulario interactivo para nuevos solicitantes",
                    "✅ Validaciones en tiempo real",
                    "🎯 Predicciones con probabilidades por clase",
                    "📊 Explicaciones de factores de riesgo",
                    "📁 Modo batch para múltiples predicciones"
                ]
            )
    
    elif page == "🔄 Re-entrenamiento":
        # Importar y renderizar el módulo de re-entrenamiento
        try:
            from src.retraining import render_retraining_module
            render_retraining_module()
        except ImportError as e:
            st.error(f"❌ Error importando el módulo de re-entrenamiento: {e}")
            render_placeholder_module(
                "🔄 Re-entrenamiento",
                "Actualización de modelos con nuevos datos",
                [
                    "📊 Detección automática de data drift",
                    "🔄 Re-entrenamiento incremental",
                    "📈 Versionado de modelos",
                    "📊 Comparación antes/después",
                    "🔙 Sistema de rollback"
                ]
            )
    
    elif page == "🎓 Aprende sobre RBMs":
        # Importar y renderizar el módulo RAG funcional
        try:
            from src.educational_rag import render_educational_rag_module
            render_educational_rag_module()
        except ImportError as e:
            st.error(f"❌ Error importando el módulo RAG educativo: {e}")
            render_placeholder_module(
                "🎓 Aprende sobre RBMs",
                "Sistema RAG educativo con papers científicos",
                [
                    "🤖 Chat interactivo con Groq AI (Llama 3.3 70B)",
                    "📚 Base de conocimiento con papers científicos",
                    "🔍 Búsqueda semántica con embeddings",
                    "📤 Carga automática de PDFs",
                    "💬 Preguntas sugeridas por nivel"
                ]
            )
    
    elif page == "📚 Documentación":
        st.title("📚 Documentación")
        
        st.markdown("""
        ## 📖 Documentación en Sphinx
        
        Toda la documentación del código, manual de uso e instalación están disponibles en Sphinx.
        
        ## 👥 Autores
        
        **Autor:**
        Andrés Fernando Gómez Rojas
        Pregrado en Física
        Universidad Distrital Francisco José de Caldas
        
        **Director:**
        Carlos Andrés Gómez Vasco
        """)

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    main()