"""
============================================================================
MÓDULO DE ANÁLISIS BIVARIADO
============================================================================

Análisis de relaciones entre pares de variables.
Incluye correlaciones, tablas de contingencia y tests estadísticos.

Autor: Sistema de Física
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')

class BivariateAnalyzer:
    """Analizador de relaciones bivariadas"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el analizador
        
        Args:
            data: DataFrame con los datos a analizar
        """
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def analyze_numeric_vs_numeric(self, var1: str, var2: str) -> Dict:
        """
        Análisis de relación entre dos variables numéricas
        
        Args:
            var1, var2: Nombres de las variables
            
        Returns:
            Diccionario con correlaciones y tests
        """
        # Filtrar datos válidos
        data_clean = self.data[[var1, var2]].dropna()
        
        if len(data_clean) < 3:
            return {'error': 'Insuficientes datos válidos'}
        
        x = data_clean[var1]
        y = data_clean[var2]
        
        # Correlaciones
        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)
        kendall_corr, kendall_p = kendalltau(x, y)
        
        # Regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        results = {
            'n_observations': len(data_clean),
            'missing_pairs': len(self.data) - len(data_clean),
            'correlations': {
                'pearson': {'r': pearson_corr, 'p_value': pearson_p},
                'spearman': {'r': spearman_corr, 'p_value': spearman_p},
                'kendall': {'r': kendall_corr, 'p_value': kendall_p}
            },
            'linear_regression': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err
            }
        }
        
        return results
    
    def analyze_categorical_vs_categorical(self, var1: str, var2: str) -> Dict:
        """
        Análisis de relación entre dos variables categóricas
        
        Args:
            var1, var2: Nombres de las variables
            
        Returns:
            Diccionario con tabla de contingencia y tests
        """
        # Crear tabla de contingencia
        contingency_table = pd.crosstab(self.data[var1], self.data[var2])
        
        # Test Chi-cuadrado
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # V de Cramér
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        results = {
            'contingency_table': contingency_table,
            'chi2_test': {
                'statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'expected_frequencies': expected
            },
            'cramers_v': cramers_v,
            'association_strength': self._interpret_cramers_v(cramers_v)
        }
        
        return results
    
    def analyze_numeric_vs_categorical(self, numeric_var: str, categorical_var: str) -> Dict:
        """
        Análisis de relación entre variable numérica y categórica
        
        Args:
            numeric_var: Variable numérica
            categorical_var: Variable categórica
            
        Returns:
            Diccionario con estadísticas por grupo y tests
        """
        # Estadísticas descriptivas por grupo
        grouped_stats = self.data.groupby(categorical_var)[numeric_var].agg([
            'count', 'mean', 'median', 'std', 'min', 'max', 'skew'
        ]).round(4)
        
        # Test ANOVA (paramétrico)
        groups = [group[numeric_var].dropna() for name, group in self.data.groupby(categorical_var)]
        
        if len(groups) >= 2 and all(len(group) >= 2 for group in groups):
            f_stat, anova_p = stats.f_oneway(*groups)
            
            # Test Kruskal-Wallis (no paramétrico)
            kw_stat, kw_p = stats.kruskal(*groups)
        else:
            f_stat = anova_p = kw_stat = kw_p = np.nan
        
        results = {
            'grouped_statistics': grouped_stats,
            'anova_test': {
                'f_statistic': f_stat,
                'p_value': anova_p
            },
            'kruskal_wallis_test': {
                'statistic': kw_stat,
                'p_value': kw_p
            },
            'groups_info': {
                'n_groups': len(groups),
                'group_sizes': [len(group) for group in groups]
            }
        }
        
        return results
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpreta el valor de V de Cramér"""
        if cramers_v < 0.1:
            return "Asociación muy débil"
        elif cramers_v < 0.3:
            return "Asociación débil"
        elif cramers_v < 0.5:
            return "Asociación moderada"
        else:
            return "Asociación fuerte"
    
    def create_correlation_matrix(self, method: str = 'pearson') -> go.Figure:
        """
        Crea matriz de correlación interactiva
        
        Args:
            method: 'pearson', 'spearman', o 'kendall'
            
        Returns:
            Figura de Plotly
        """
        # Calcular matriz de correlación
        if method == 'pearson':
            corr_matrix = self.data[self.numeric_columns].corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = self.data[self.numeric_columns].corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = self.data[self.numeric_columns].corr(method='kendall')
        
        # Crear heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True,
            colorbar=dict(title=f"Correlación {method.title()}")
        ))
        
        fig.update_layout(
            title=f"Matriz de Correlación - {method.title()}",
            template="plotly_white",
            height=600,
            width=800
        )
        
        return fig
    
    def create_scatter_plot(self, var1: str, var2: str, color_var: str = None) -> go.Figure:
        """
        Crea gráfico de dispersión con línea de regresión
        
        Args:
            var1, var2: Variables para ejes X e Y
            color_var: Variable para colorear puntos (opcional)
            
        Returns:
            Figura de Plotly
        """
        # Filtrar datos válidos
        if color_var:
            data_clean = self.data[[var1, var2, color_var]].dropna()
        else:
            data_clean = self.data[[var1, var2]].dropna()
        
        # Crear scatter plot
        if color_var and color_var in self.categorical_columns:
            fig = px.scatter(
                data_clean,
                x=var1,
                y=var2,
                color=color_var,
                trendline="ols",
                title=f"Relación: {var1} vs {var2}",
                labels={var1: var1, var2: var2}
            )
        else:
            fig = px.scatter(
                data_clean,
                x=var1,
                y=var2,
                trendline="ols",
                title=f"Relación: {var1} vs {var2}",
                labels={var1: var1, var2: var2}
            )
        
        fig.update_layout(
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_contingency_heatmap(self, var1: str, var2: str) -> go.Figure:
        """
        Crea heatmap de tabla de contingencia
        
        Args:
            var1, var2: Variables categóricas
            
        Returns:
            Figura de Plotly
        """
        # Crear tabla de contingencia
        contingency = pd.crosstab(self.data[var1], self.data[var2])
        
        # Crear heatmap
        fig = go.Figure(data=go.Heatmap(
            z=contingency.values,
            x=contingency.columns,
            y=contingency.index,
            colorscale='Blues',
            text=contingency.values,
            texttemplate='%{text}',
            textfont={"size": 12},
            showscale=True,
            colorbar=dict(title="Frecuencia")
        ))
        
        fig.update_layout(
            title=f"Tabla de Contingencia: {var1} vs {var2}",
            xaxis_title=var2,
            yaxis_title=var1,
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_grouped_boxplot(self, numeric_var: str, categorical_var: str) -> go.Figure:
        """
        Crea boxplot agrupado
        
        Args:
            numeric_var: Variable numérica
            categorical_var: Variable categórica
            
        Returns:
            Figura de Plotly
        """
        fig = px.box(
            self.data,
            x=categorical_var,
            y=numeric_var,
            title=f"Distribución de {numeric_var} por {categorical_var}",
            color=categorical_var
        )
        
        fig.update_layout(
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        return fig

def render_bivariate_analysis():
    """Renderiza el módulo de análisis bivariado en Streamlit"""
    st.title("📊 Análisis Bivariado")
    st.markdown("### *Análisis de relaciones entre pares de variables*")
    
    # Verificar datos
    if not os.path.exists("data/processed/datos_credito_hipotecario_realista.csv"):
        st.error("❌ No hay datos disponibles. Ve a 'Generar Datos' primero.")
        return
    
    # Cargar datos
    @st.cache_data
    def load_data():
        return pd.read_csv("data/processed/datos_credito_hipotecario_realista.csv")
    
    df = load_data()
    st.success(f"✅ Datos cargados: {len(df):,} registros, {len(df.columns)} variables")
    
    # Crear analizador
    analyzer = BivariateAnalyzer(df)
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔗 Correlaciones",
        "📈 Numérica vs Numérica", 
        "📊 Categórica vs Categórica",
        "📦 Numérica vs Categórica"
    ])
    
    # ==================== TAB 1: CORRELACIONES ====================
    with tab1:
        st.subheader("🔗 Matriz de Correlación")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            correlation_method = st.selectbox(
                "Método de correlación:",
                options=['pearson', 'spearman', 'kendall'],
                help="Pearson: lineal, Spearman: monotónica, Kendall: ordinal"
            )
            
            min_corr = st.slider(
                "Correlación mínima a mostrar:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Filtrar correlaciones débiles"
            )
        
        with col2:
            if st.button("📊 Generar Matriz de Correlación", type="primary"):
                with st.spinner("📊 Calculando correlaciones..."):
                    fig_corr = analyzer.create_correlation_matrix(correlation_method)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Tabla de correlaciones más altas
                    corr_matrix = df[analyzer.numeric_columns].corr(method=correlation_method)
                    
                    # Extraer correlaciones únicas
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    corr_pairs = corr_matrix.where(mask).stack().reset_index()
                    corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlación']
                    corr_pairs['Correlación Abs'] = abs(corr_pairs['Correlación'])
                    
                    # Filtrar y ordenar
                    high_corr = corr_pairs[
                        corr_pairs['Correlación Abs'] >= min_corr
                    ].sort_values('Correlación Abs', ascending=False)
                    
                    st.subheader(f"🔝 Top Correlaciones (|r| ≥ {min_corr})")
                    st.dataframe(
                        high_corr.head(20)[['Variable 1', 'Variable 2', 'Correlación']],
                        use_container_width=True,
                        hide_index=True
                    )
    
    # ==================== TAB 2: NUMÉRICA VS NUMÉRICA ====================
    with tab2:
        st.subheader("📈 Análisis: Numérica vs Numérica")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var1 = st.selectbox(
                "Variable X:",
                options=analyzer.numeric_columns,
                key="num_var1"
            )
        
        with col2:
            var2 = st.selectbox(
                "Variable Y:",
                options=analyzer.numeric_columns,
                key="num_var2"
            )
        
        with col3:
            color_var = st.selectbox(
                "Colorear por (opcional):",
                options=[None] + analyzer.categorical_columns,
                key="color_var"
            )
        
        if var1 and var2 and var1 != var2:
            # Análisis estadístico
            results = analyzer.analyze_numeric_vs_numeric(var1, var2)
            
            if 'error' not in results:
                # Mostrar métricas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Correlación Pearson", f"{results['correlations']['pearson']['r']:.3f}")
                
                with col2:
                    st.metric("Correlación Spearman", f"{results['correlations']['spearman']['r']:.3f}")
                
                with col3:
                    st.metric("R² (Regresión)", f"{results['linear_regression']['r_squared']:.3f}")
                
                with col4:
                    st.metric("Observaciones", f"{results['n_observations']:,}")
                
                # Gráfico de dispersión
                fig_scatter = analyzer.create_scatter_plot(var1, var2, color_var)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Interpretación
                pearson_r = results['correlations']['pearson']['r']
                if abs(pearson_r) < 0.3:
                    strength = "débil"
                elif abs(pearson_r) < 0.7:
                    strength = "moderada"
                else:
                    strength = "fuerte"
                
                direction = "positiva" if pearson_r > 0 else "negativa"
                
                st.info(f"💡 **Interpretación:** Correlación {strength} {direction} (r = {pearson_r:.3f})")
            else:
                st.error(results['error'])
    
    # ==================== TAB 3: CATEGÓRICA VS CATEGÓRICA ====================
    with tab3:
        st.subheader("📊 Análisis: Categórica vs Categórica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cat_var1 = st.selectbox(
                "Variable 1:",
                options=analyzer.categorical_columns,
                key="cat_var1"
            )
        
        with col2:
            cat_var2 = st.selectbox(
                "Variable 2:",
                options=analyzer.categorical_columns,
                key="cat_var2"
            )
        
        if cat_var1 and cat_var2 and cat_var1 != cat_var2:
            # Análisis estadístico
            results = analyzer.analyze_categorical_vs_categorical(cat_var1, cat_var2)
            
            # Mostrar métricas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Chi² Estadístico", f"{results['chi2_test']['statistic']:.3f}")
            
            with col2:
                st.metric("P-valor", f"{results['chi2_test']['p_value']:.6f}")
            
            with col3:
                st.metric("V de Cramér", f"{results['cramers_v']:.3f}")
            
            # Interpretación
            if results['chi2_test']['p_value'] < 0.05:
                st.success("✅ Existe asociación significativa (p < 0.05)")
            else:
                st.warning("⚠️ No hay asociación significativa (p ≥ 0.05)")
            
            st.info(f"💡 **Fuerza de asociación:** {results['association_strength']}")
            
            # Visualizaciones
            col1, col2 = st.columns(2)
            
            with col1:
                # Heatmap de contingencia
                fig_heatmap = analyzer.create_contingency_heatmap(cat_var1, cat_var2)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                # Gráfico de barras apiladas
                contingency = results['contingency_table']
                fig_stacked = go.Figure()
                
                for col in contingency.columns:
                    fig_stacked.add_trace(go.Bar(
                        name=str(col),
                        x=contingency.index,
                        y=contingency[col]
                    ))
                
                fig_stacked.update_layout(
                    title=f"Distribución: {cat_var1} vs {cat_var2}",
                    xaxis_title=cat_var1,
                    yaxis_title="Frecuencia",
                    barmode='stack',
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig_stacked, use_container_width=True)
            
            # Tabla de contingencia
            st.subheader("📋 Tabla de Contingencia")
            st.dataframe(results['contingency_table'], use_container_width=True)
    
    # ==================== TAB 4: NUMÉRICA VS CATEGÓRICA ====================
    with tab4:
        st.subheader("📦 Análisis: Numérica vs Categórica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_var = st.selectbox(
                "Variable Numérica:",
                options=analyzer.numeric_columns,
                key="num_var_mixed"
            )
        
        with col2:
            cat_var = st.selectbox(
                "Variable Categórica:",
                options=analyzer.categorical_columns,
                key="cat_var_mixed"
            )
        
        if num_var and cat_var:
            # Análisis estadístico
            results = analyzer.analyze_numeric_vs_categorical(num_var, cat_var)
            
            # Mostrar métricas de tests
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("F-estadístico (ANOVA)", f"{results['anova_test']['f_statistic']:.3f}")
            
            with col2:
                st.metric("P-valor ANOVA", f"{results['anova_test']['p_value']:.6f}")
            
            with col3:
                st.metric("Grupos", results['groups_info']['n_groups'])
            
            # Interpretación de tests
            if results['anova_test']['p_value'] < 0.05:
                st.success("✅ Diferencias significativas entre grupos (ANOVA p < 0.05)")
            else:
                st.warning("⚠️ No hay diferencias significativas entre grupos (ANOVA p ≥ 0.05)")
            
            # Visualizaciones
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot agrupado
                fig_box = analyzer.create_grouped_boxplot(num_var, cat_var)
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Violin plot
                fig_violin = px.violin(
                    df,
                    x=cat_var,
                    y=num_var,
                    title=f"Distribución de {num_var} por {cat_var}",
                    box=True
                )
                fig_violin.update_layout(
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            
            # Estadísticas por grupo
            st.subheader("📊 Estadísticas por Grupo")
            st.dataframe(results['grouped_statistics'], use_container_width=True)

def render_bivariate_module():
    """Función principal para renderizar el módulo bivariado"""
    render_bivariate_analysis()

if __name__ == "__main__":
    print("Módulo de análisis bivariado cargado correctamente")