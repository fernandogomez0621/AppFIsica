
"""
============================================================================
MÓDULO DE CLUSTERING
============================================================================

Segmentación de solicitantes en grupos homogéneos usando múltiples algoritmos
de clustering con visualizaciones PCA 2D/3D interactivas.

Autor: Sistema de Física
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')

class ClusterAnalyzer:
    """Analizador de clustering para segmentación de clientes"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el analizador de clustering
        
        Args:
            data: DataFrame con datos a analizar
        """
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler = StandardScaler()
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.pca_3d = PCA(n_components=3, random_state=42)
        
        # Resultados de clustering
        self.clustering_results = {}
        self.optimal_k = None
        
    def prepare_data(self, selected_features: List[str]) -> np.ndarray:
        """
        Prepara datos para clustering
        
        Args:
            selected_features: Lista de características seleccionadas
            
        Returns:
            Datos escalados
        """
        # Filtrar características numéricas válidas
        valid_features = [f for f in selected_features if f in self.numeric_columns]
        
        if not valid_features:
            raise ValueError("No hay características numéricas válidas seleccionadas")
        
        # Extraer datos y manejar valores faltantes
        X = self.data[valid_features].fillna(self.data[valid_features].median())
        
        # Escalar datos
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, valid_features
    
    def find_optimal_k(self, X: np.ndarray, max_k: int = 10) -> Dict:
        """
        Encuentra el número óptimo de clusters usando múltiples métodos
        
        Args:
            X: Datos escalados
            max_k: Número máximo de clusters a evaluar
            
        Returns:
            Diccionario con métricas por k
        """
        k_range = range(2, max_k + 1)
        
        # Métricas para cada k
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        for k in k_range:
            # K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Métricas
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            davies_bouldin_scores.append(davies_bouldin_score(X, labels))
            calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
        
        # Encontrar k óptimo por cada método
        # Método del codo (buscar el "codo" en la curva de inercia)
        elbow_k = self._find_elbow_point(list(k_range), inertias)
        
        # Mejor silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Menor Davies-Bouldin score
        best_db_k = k_range[np.argmin(davies_bouldin_scores)]
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'calinski_harabasz_scores': calinski_harabasz_scores,
            'optimal_k_methods': {
                'elbow': elbow_k,
                'silhouette': best_silhouette_k,
                'davies_bouldin': best_db_k
            }
        }
        
        return results
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Encuentra el punto del codo en la curva de inercia"""
        # Método de la segunda derivada
        if len(inertias) < 3:
            return k_values[0]
        
        # Calcular diferencias
        first_diff = np.diff(inertias)
        second_diff = np.diff(first_diff)
        
        # Encontrar el punto con mayor segunda derivada (más curvatura)
        elbow_idx = np.argmax(second_diff) + 2  # +2 porque perdemos 2 puntos en las diferencias
        
        if elbow_idx < len(k_values):
            return k_values[elbow_idx]
        else:
            return k_values[len(k_values)//2]  # Fallback al punto medio
    
    def perform_clustering(self, X: np.ndarray, algorithm: str, n_clusters: int, **kwargs) -> Dict:
        """
        Ejecuta algoritmo de clustering
        
        Args:
            X: Datos escalados
            algorithm: 'kmeans', 'hierarchical', 'dbscan', 'gmm'
            n_clusters: Número de clusters
            **kwargs: Parámetros adicionales del algoritmo
            
        Returns:
            Resultados del clustering
        """
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            centers = model.cluster_centers_
            
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
            centers = None
            
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            centers = None
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
        elif algorithm == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            centers = model.means_
        
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")
        
        # Calcular métricas si hay más de 1 cluster
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
        else:
            silhouette = davies_bouldin = calinski_harabasz = np.nan
        
        results = {
            'model': model,
            'labels': labels,
            'centers': centers,
            'n_clusters': n_clusters,
            'metrics': {
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz
            },
            'cluster_sizes': pd.Series(labels).value_counts().sort_index().to_dict()
        }
        
        return results
    
    def create_pca_visualizations(self, X: np.ndarray, labels: np.ndarray, 
                                feature_names: List[str]) -> Dict:
        """
        Crea visualizaciones PCA 2D y 3D
        
        Args:
            X: Datos escalados
            labels: Etiquetas de cluster
            feature_names: Nombres de características
            
        Returns:
            Diccionario con figuras
        """
        figures = {}
        
        # PCA 2D
        X_pca_2d = self.pca_2d.fit_transform(X)
        explained_variance_2d = self.pca_2d.explained_variance_ratio_
        
        # Crear DataFrame para PCA 2D
        pca_2d_df = pd.DataFrame({
            'PC1': X_pca_2d[:, 0],
            'PC2': X_pca_2d[:, 1],
            'Cluster': labels.astype(str)
        })
        
        # Gráfico PCA 2D
        fig_2d = px.scatter(
            pca_2d_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=f"Clustering - PCA 2D<br>PC1: {explained_variance_2d[0]:.1%} varianza, PC2: {explained_variance_2d[1]:.1%} varianza",
            labels={
                'PC1': f'PC1 ({explained_variance_2d[0]:.1%} varianza)',
                'PC2': f'PC2 ({explained_variance_2d[1]:.1%} varianza)'
            }
        )
        
        fig_2d.update_layout(
            template="plotly_white",
            height=500
        )
        
        figures['pca_2d'] = fig_2d
        
        # PCA 3D
        X_pca_3d = self.pca_3d.fit_transform(X)
        explained_variance_3d = self.pca_3d.explained_variance_ratio_
        
        # Crear DataFrame para PCA 3D
        pca_3d_df = pd.DataFrame({
            'PC1': X_pca_3d[:, 0],
            'PC2': X_pca_3d[:, 1],
            'PC3': X_pca_3d[:, 2],
            'Cluster': labels.astype(str)
        })
        
        # Gráfico PCA 3D
        fig_3d = px.scatter_3d(
            pca_3d_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Cluster',
            title=f"Clustering - PCA 3D<br>Varianza explicada: PC1={explained_variance_3d[0]:.1%}, PC2={explained_variance_3d[1]:.1%}, PC3={explained_variance_3d[2]:.1%}",
            labels={
                'PC1': f'PC1 ({explained_variance_3d[0]:.1%})',
                'PC2': f'PC2 ({explained_variance_3d[1]:.1%})',
                'PC3': f'PC3 ({explained_variance_3d[2]:.1%})'
            }
        )
        
        fig_3d.update_layout(
            template="plotly_white",
            height=600
        )
        
        figures['pca_3d'] = fig_3d
        
        return figures
    
    def analyze_clusters(self, df_with_clusters: pd.DataFrame,
                        feature_names: List[str]) -> Dict:
        """
        Analiza perfiles de clusters con estadísticas descriptivas completas
        
        Args:
            df_with_clusters: DataFrame con columna 'cluster'
            feature_names: Nombres de características usadas
            
        Returns:
            Análisis por cluster
        """
        cluster_analysis = {}
        
        # Análisis por cluster
        for cluster_id in sorted(df_with_clusters['cluster'].unique()):
            if cluster_id == -1:  # Outliers en DBSCAN
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Estadísticas numéricas completas para características usadas
            numeric_stats = cluster_data[feature_names].describe()
            
            # Estadísticas adicionales (percentiles, desviación estándar, coeficiente de variación)
            additional_stats = {}
            for feature in feature_names:
                if feature in cluster_data.columns:
                    data = cluster_data[feature].dropna()
                    if len(data) > 0:
                        additional_stats[feature] = {
                            'median': data.median(),
                            'std': data.std(),
                            'cv': (data.std() / data.mean() * 100) if data.mean() != 0 else 0,
                            'q1': data.quantile(0.25),
                            'q3': data.quantile(0.75),
                            'iqr': data.quantile(0.75) - data.quantile(0.25),
                            'skewness': data.skew(),
                            'kurtosis': data.kurtosis()
                        }
            
            # Estadísticas de variables principales del dataset completo
            main_variables = ['edad', 'salario_mensual', 'puntaje_datacredito', 'dti_ratio',
                            'monto_solicitado', 'plazo_meses', 'cuota_mensual',
                            'antiguedad_laboral_meses', 'num_dependientes']
            
            main_stats = {}
            for var in main_variables:
                if var in cluster_data.columns:
                    data = cluster_data[var].dropna()
                    if len(data) > 0:
                        main_stats[var] = {
                            'count': len(data),
                            'mean': data.mean(),
                            'median': data.median(),
                            'std': data.std(),
                            'min': data.min(),
                            'max': data.max(),
                            'q1': data.quantile(0.25),
                            'q3': data.quantile(0.75),
                            'cv': (data.std() / data.mean() * 100) if data.mean() != 0 else 0,
                            'skewness': data.skew(),
                            'kurtosis': data.kurtosis()
                        }
            
            # Distribución de riesgo si existe
            risk_distribution = {}
            if 'nivel_riesgo' in cluster_data.columns:
                risk_counts = cluster_data['nivel_riesgo'].value_counts()
                risk_distribution = (risk_counts / len(cluster_data) * 100).to_dict()
            
            # Distribución de variables categóricas relevantes
            categorical_distributions = {}
            categorical_vars = ['tipo_vivienda', 'tipo_empleo', 'nivel_educacion',
                              'estado_civil', 'genero']
            
            for var in categorical_vars:
                if var in cluster_data.columns:
                    counts = cluster_data[var].value_counts()
                    categorical_distributions[var] = (counts / len(cluster_data) * 100).to_dict()
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100,
                'numeric_stats': numeric_stats,
                'additional_stats': additional_stats,
                'main_stats': main_stats,
                'risk_distribution': risk_distribution,
                'categorical_distributions': categorical_distributions
            }
        
        return cluster_analysis

def render_clustering_analysis():
    """Renderiza el módulo de clustering en Streamlit"""
    st.title("🎯 Análisis de Clustering")
    st.markdown("### *Segmentación de solicitantes en grupos homogéneos*")
    
    # Verificar datos
    data_path = "data/processed/datos_con_caracteristicas.csv"
    if not os.path.exists(data_path):
        # Intentar con datos originales
        data_path = "data/processed/datos_credito_hipotecario_realista.csv"
        if not os.path.exists(data_path):
            st.error("❌ No hay datos disponibles. Ve a 'Generar Datos' primero.")
            return
        else:
            st.warning("⚠️ Usando datos originales. Se recomienda crear características primero en 'Ingeniería de Características'.")
    
    # Cargar datos
    @st.cache_data
    def load_data():
        return pd.read_csv(data_path)
    
    df = load_data()
    st.success(f"✅ Datos cargados: {len(df):,} registros, {len(df.columns)} variables")
    
    # Crear analizador
    analyzer = ClusterAnalyzer(df)
    
    # Configuración de clustering
    st.subheader("⚙️ Configuración del Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Selección de Características:**")
        
        # Filtrar características numéricas relevantes
        relevant_features = [col for col in analyzer.numeric_columns 
                           if not col.endswith('_cod') and col != 'rechazo_automatico']
        
        selected_features = st.multiselect(
            "Características para clustering:",
            options=relevant_features,
            default=relevant_features[:10] if len(relevant_features) > 10 else relevant_features,
            help="Selecciona las variables para el análisis de clustering"
        )
    
    with col2:
        st.markdown("**Parámetros del Algoritmo:**")
        
        algorithm = st.selectbox(
            "Algoritmo de clustering:",
            options=['kmeans', 'hierarchical', 'dbscan', 'gmm'],
            format_func=lambda x: {
                'kmeans': 'K-Means',
                'hierarchical': 'Jerárquico',
                'dbscan': 'DBSCAN',
                'gmm': 'Gaussian Mixture'
            }[x]
        )
        
        if algorithm in ['kmeans', 'hierarchical', 'gmm']:
            n_clusters = st.slider(
                "Número de clusters:",
                min_value=2,
                max_value=10,
                value=3,
                help="Número de grupos a formar"
            )
        else:  # DBSCAN
            eps = st.slider("Epsilon (eps):", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min samples:", 2, 20, 5)
            n_clusters = None
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una característica.")
        return
    
    # Preparar datos
    try:
        X_scaled, valid_features = analyzer.prepare_data(selected_features)
        st.info(f"📊 Datos preparados: {X_scaled.shape[0]} muestras × {X_scaled.shape[1]} características")
    except Exception as e:
        st.error(f"❌ Error preparando datos: {e}")
        return
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs([
        "🔍 Optimización de K",
        "🎯 Clustering",
        "📊 Análisis de Clusters"
    ])
    
    # ==================== TAB 1: OPTIMIZACIÓN DE K ====================
    with tab1:
        st.subheader("🔍 Determinación del Número Óptimo de Clusters")
        
        max_k = st.slider("Máximo K a evaluar:", 3, 15, 10)
        
        if st.button("📊 Evaluar K Óptimo", type="primary"):
            with st.spinner("🔍 Evaluando diferentes valores de K..."):
                k_results = analyzer.find_optimal_k(X_scaled, max_k)
                
                # Crear gráficos de evaluación
                fig_metrics = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        "Método del Codo (Inercia)",
                        "Coeficiente de Silueta",
                        "Índice Davies-Bouldin",
                        "Índice Calinski-Harabasz"
                    ],
                    vertical_spacing=0.1
                )
                
                # Método del codo
                fig_metrics.add_trace(
                    go.Scatter(
                        x=k_results['k_range'],
                        y=k_results['inertias'],
                        mode='lines+markers',
                        name='Inercia',
                        line=dict(color='#e74c3c')
                    ),
                    row=1, col=1
                )
                
                # Silhouette score
                fig_metrics.add_trace(
                    go.Scatter(
                        x=k_results['k_range'],
                        y=k_results['silhouette_scores'],
                        mode='lines+markers',
                        name='Silhouette',
                        line=dict(color='#3498db')
                    ),
                    row=1, col=2
                )
                
                # Davies-Bouldin
                fig_metrics.add_trace(
                    go.Scatter(
                        x=k_results['k_range'],
                        y=k_results['davies_bouldin_scores'],
                        mode='lines+markers',
                        name='Davies-Bouldin',
                        line=dict(color='#f39c12')
                    ),
                    row=2, col=1
                )
                
                # Calinski-Harabasz
                fig_metrics.add_trace(
                    go.Scatter(
                        x=k_results['k_range'],
                        y=k_results['calinski_harabasz_scores'],
                        mode='lines+markers',
                        name='Calinski-Harabasz',
                        line=dict(color='#9b59b6')
                    ),
                    row=2, col=2
                )
                
                fig_metrics.update_layout(
                    title="Métricas de Evaluación de Clustering",
                    template="plotly_white",
                    height=600,
                    showlegend=False
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
                
                # Mostrar recomendaciones
                st.subheader("🎯 Recomendaciones de K Óptimo")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Método del Codo", k_results['optimal_k_methods']['elbow'])
                
                with col2:
                    st.metric("Mejor Silhouette", k_results['optimal_k_methods']['silhouette'])
                
                with col3:
                    st.metric("Mejor Davies-Bouldin", k_results['optimal_k_methods']['davies_bouldin'])
                
                # Guardar resultados en session state
                st.session_state.k_optimization_results = k_results
    
    # ==================== TAB 2: CLUSTERING ====================
    with tab2:
        st.subheader("🎯 Ejecutar Clustering")
        
        # Usar K recomendado si está disponible
        if 'k_optimization_results' in st.session_state and algorithm in ['kmeans', 'hierarchical', 'gmm']:
            recommended_k = st.session_state.k_optimization_results['optimal_k_methods']['silhouette']
            st.info(f"💡 K recomendado por Silhouette: {recommended_k}")
        
        # Botón para ejecutar clustering
        if algorithm == 'dbscan':
            button_text = f"🎯 Ejecutar {algorithm.upper()} (eps={eps}, min_samples={min_samples})"
            clustering_params = {'eps': eps, 'min_samples': min_samples}
        else:
            button_text = f"🎯 Ejecutar {algorithm.upper()} (k={n_clusters})"
            clustering_params = {}
        
        if st.button(button_text, type="primary"):
            with st.spinner(f"🎯 Ejecutando {algorithm.upper()}..."):
                try:
                    # Ejecutar clustering
                    clustering_results = analyzer.perform_clustering(
                        X_scaled, algorithm, n_clusters, **clustering_params
                    )
                    
                    # Guardar resultados
                    st.session_state.clustering_results = clustering_results
                    st.session_state.clustering_features = valid_features
                    st.session_state.clustering_data = X_scaled
                    
                    # Mostrar métricas
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Clusters Formados", clustering_results['n_clusters'])
                    
                    with col2:
                        if not np.isnan(clustering_results['metrics']['silhouette_score']):
                            st.metric("Silhouette Score", f"{clustering_results['metrics']['silhouette_score']:.3f}")
                    
                    with col3:
                        if not np.isnan(clustering_results['metrics']['davies_bouldin_score']):
                            st.metric("Davies-Bouldin", f"{clustering_results['metrics']['davies_bouldin_score']:.3f}")
                    
                    with col4:
                        if not np.isnan(clustering_results['metrics']['calinski_harabasz_score']):
                            st.metric("Calinski-Harabasz", f"{clustering_results['metrics']['calinski_harabasz_score']:.0f}")
                    
                    # Crear visualizaciones PCA
                    pca_figures = analyzer.create_pca_visualizations(
                        X_scaled, clustering_results['labels'], valid_features
                    )
                    
                    # Mostrar visualizaciones
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(pca_figures['pca_2d'], use_container_width=True)
                    
                    with col2:
                        # Gráfico de tamaños de cluster
                        cluster_sizes = clustering_results['cluster_sizes']
                        
                        fig_sizes = px.bar(
                            x=list(cluster_sizes.keys()),
                            y=list(cluster_sizes.values()),
                            title="Tamaño de Clusters",
                            labels={'x': 'Cluster', 'y': 'Número de Muestras'}
                        )
                        
                        fig_sizes.update_layout(
                            template="plotly_white",
                            height=400
                        )
                        
                        st.plotly_chart(fig_sizes, use_container_width=True)
                    
                    # Visualización 3D
                    st.plotly_chart(pca_figures['pca_3d'], use_container_width=True)
                    
                    st.success("✅ Clustering ejecutado exitosamente!")
                    
                except Exception as e:
                    st.error(f"❌ Error ejecutando clustering: {e}")
                    st.exception(e)
    
    # ==================== TAB 3: ANÁLISIS DE CLUSTERS ====================
    with tab3:
        st.subheader("📊 Análisis de Clusters")
        
        if 'clustering_results' not in st.session_state:
            st.info("ℹ️ Ejecuta clustering primero en la pestaña anterior.")
            return
        
        clustering_results = st.session_state.clustering_results
        valid_features = st.session_state.clustering_features
        
        # Agregar etiquetas de cluster al DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clustering_results['labels']
        
        # Análisis de clusters
        cluster_analysis = analyzer.analyze_clusters(df_with_clusters, valid_features)
        
        # Mostrar análisis por cluster
        for cluster_id, analysis in cluster_analysis.items():
            with st.expander(f"🎯 Cluster {cluster_id} - {analysis['size']} muestras ({analysis['percentage']:.1f}%)", expanded=True):
                
                # Tabs para organizar la información
                tab_stats, tab_dist, tab_cat = st.tabs([
                    "📊 Estadísticas Completas",
                    "📈 Distribuciones",
                    "🏷️ Variables Categóricas"
                ])
                
                # ========== TAB: ESTADÍSTICAS COMPLETAS ==========
                with tab_stats:
                    st.markdown("### 📊 Estadísticas Descriptivas de Variables Principales")
                    
                    if analysis['main_stats']:
                        # Crear DataFrame con todas las estadísticas
                        stats_data = []
                        for var, stats in analysis['main_stats'].items():
                            stats_data.append({
                                'Variable': var,
                                'N': int(stats['count']),
                                'Media': f"{stats['mean']:.2f}",
                                'Mediana': f"{stats['median']:.2f}",
                                'Desv.Est': f"{stats['std']:.2f}",
                                'CV%': f"{stats['cv']:.1f}",
                                'Mín': f"{stats['min']:.2f}",
                                'Q1': f"{stats['q1']:.2f}",
                                'Q3': f"{stats['q3']:.2f}",
                                'Máx': f"{stats['max']:.2f}",
                                'Asimetría': f"{stats['skewness']:.2f}",
                                'Curtosis': f"{stats['kurtosis']:.2f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        # Interpretación de estadísticas clave
                        st.markdown("#### 🔍 Interpretación de Métricas:")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("""
                            **Coeficiente de Variación (CV%)**
                            - < 15%: Baja variabilidad
                            - 15-30%: Variabilidad moderada
                            - > 30%: Alta variabilidad
                            """)
                        
                        with col2:
                            st.markdown("""
                            **Asimetría**
                            - ≈ 0: Distribución simétrica
                            - > 0: Sesgo a la derecha
                            - < 0: Sesgo a la izquierda
                            """)
                        
                        with col3:
                            st.markdown("""
                            **Curtosis**
                            - ≈ 0: Distribución normal
                            - > 0: Más puntiaguda
                            - < 0: Más aplanada
                            """)
                    
                    # Estadísticas de características usadas en clustering
                    if analysis['additional_stats']:
                        st.markdown("### 📐 Estadísticas de Características de Clustering")
                        
                        add_stats_data = []
                        for feature, stats in analysis['additional_stats'].items():
                            add_stats_data.append({
                                'Característica': feature,
                                'Mediana': f"{stats['median']:.3f}",
                                'Desv.Est': f"{stats['std']:.3f}",
                                'CV%': f"{stats['cv']:.1f}",
                                'Q1': f"{stats['q1']:.3f}",
                                'Q3': f"{stats['q3']:.3f}",
                                'IQR': f"{stats['iqr']:.3f}",
                                'Asimetría': f"{stats['skewness']:.2f}",
                                'Curtosis': f"{stats['kurtosis']:.2f}"
                            })
                        
                        add_stats_df = pd.DataFrame(add_stats_data)
                        st.dataframe(add_stats_df, use_container_width=True, hide_index=True)
                
                # ========== TAB: DISTRIBUCIONES ==========
                with tab_dist:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🎯 Distribución de Riesgo")
                        
                        if analysis['risk_distribution']:
                            risk_df = pd.DataFrame([
                                [nivel, f"{porcentaje:.1f}%"]
                                for nivel, porcentaje in analysis['risk_distribution'].items()
                            ], columns=["Nivel de Riesgo", "Porcentaje"])
                            
                            st.dataframe(risk_df, use_container_width=True, hide_index=True)
                            
                            # Gráfico de torta
                            fig_risk = px.pie(
                                values=list(analysis['risk_distribution'].values()),
                                names=list(analysis['risk_distribution'].keys()),
                                title=f"Distribución de Riesgo - Cluster {cluster_id}",
                                color_discrete_map={
                                    'Bajo': '#28a745',
                                    'Medio': '#ffc107',
                                    'Alto': '#dc3545'
                                }
                            )
                            fig_risk.update_layout(height=350)
                            st.plotly_chart(fig_risk, use_container_width=True)
                        else:
                            st.info("No hay información de riesgo disponible")
                    
                    with col2:
                        st.markdown("### 📊 Comparación de Medias")
                        
                        # Gráfico de barras con las principales variables
                        if analysis['main_stats']:
                            comparison_vars = ['edad', 'salario_mensual', 'puntaje_datacredito', 'dti_ratio']
                            comparison_vars = [v for v in comparison_vars if v in analysis['main_stats']]
                            
                            if comparison_vars:
                                means = [analysis['main_stats'][v]['mean'] for v in comparison_vars]
                                
                                fig_means = go.Figure(data=[
                                    go.Bar(
                                        x=comparison_vars,
                                        y=means,
                                        text=[f"{m:.1f}" for m in means],
                                        textposition='auto',
                                        marker_color='#3498db'
                                    )
                                ])
                                
                                fig_means.update_layout(
                                    title=f"Valores Medios - Cluster {cluster_id}",
                                    xaxis_title="Variable",
                                    yaxis_title="Valor Medio",
                                    template="plotly_white",
                                    height=350
                                )
                                
                                st.plotly_chart(fig_means, use_container_width=True)
                
                # ========== TAB: VARIABLES CATEGÓRICAS ==========
                with tab_cat:
                    st.markdown("### 🏷️ Distribución de Variables Categóricas")
                    
                    if analysis['categorical_distributions']:
                        # Mostrar cada variable categórica
                        for var_name, distribution in analysis['categorical_distributions'].items():
                            if distribution:
                                st.markdown(f"#### {var_name.replace('_', ' ').title()}")
                                
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    # Tabla de distribución
                                    dist_df = pd.DataFrame([
                                        [cat, f"{pct:.1f}%"]
                                        for cat, pct in sorted(distribution.items(),
                                                              key=lambda x: x[1],
                                                              reverse=True)
                                    ], columns=["Categoría", "Porcentaje"])
                                    
                                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
                                
                                with col2:
                                    # Gráfico de barras
                                    fig_cat = px.bar(
                                        x=list(distribution.keys()),
                                        y=list(distribution.values()),
                                        title=f"{var_name.replace('_', ' ').title()}",
                                        labels={'x': 'Categoría', 'y': 'Porcentaje (%)'}
                                    )
                                    
                                    fig_cat.update_layout(
                                        template="plotly_white",
                                        height=300,
                                        showlegend=False
                                    )
                                    
                                    st.plotly_chart(fig_cat, use_container_width=True)
                                
                                st.markdown("---")
                    else:
                        st.info("No hay variables categóricas disponibles para análisis")
        
        # Guardar resultados de clustering
        if st.button("💾 Guardar Resultados de Clustering"):
            try:
                # Guardar dataset con clusters
                cluster_path = "data/processed/datos_con_clusters.csv"
                df_with_clusters.to_csv(cluster_path, index=False)
                
                st.success(f"✅ Resultados guardados: {cluster_path}")
                
            except Exception as e:
                st.error(f"❌ Error guardando: {e}")

def render_clustering_module():
    """Función principal para renderizar el módulo de clustering"""
    render_clustering_analysis()

if __name__ == "__main__":
    print("Módulo de clustering cargado correctamente")