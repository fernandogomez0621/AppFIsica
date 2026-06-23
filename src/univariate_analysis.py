"""
============================================================================
MÓDULO DE ANÁLISIS UNIVARIADO
============================================================================

Análisis estadístico descriptivo por variable individual.
Incluye estadísticas completas y visualizaciones interactivas.

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
from typing import Dict, List, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')

class UnivariateAnalyzer:
    """Analizador de estadísticas univariadas"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el analizador
        
        Args:
            data: DataFrame con los datos a analizar
        """
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def analyze_numeric_variable(self, column: str) -> Dict:
        """
        Análisis completo de variable numérica
        
        Args:
            column: Nombre de la columna a analizar
            
        Returns:
            Diccionario con estadísticas y tests
        """
        if column not in self.numeric_columns:
            raise ValueError(f"'{column}' no es una variable numérica")
        
        data = self.data[column].dropna()
        
        # Estadísticas descriptivas
        stats_dict = {
            'count': len(data),
            'missing': self.data[column].isnull().sum(),
            'missing_pct': (self.data[column].isnull().sum() / len(self.data)) * 100,
            'mean': data.mean(),
            'median': data.median(),
            'mode': data.mode().iloc[0] if len(data.mode()) > 0 else np.nan,
            'std': data.std(),
            'variance': data.var(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'cv': (data.std() / data.mean()) * 100 if data.mean() != 0 else np.inf,
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'p5': data.quantile(0.05),
            'p95': data.quantile(0.95)
        }
        
        # Tests estadísticos
        # Test de normalidad (Shapiro-Wilk para n < 5000, Kolmogorov-Smirnov para n >= 5000)
        if len(data) < 5000:
            normality_stat, normality_p = stats.shapiro(data)
            normality_test = "Shapiro-Wilk"
        else:
            normality_stat, normality_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            normality_test = "Kolmogorov-Smirnov"
        
        stats_dict.update({
            'normality_test': normality_test,
            'normality_statistic': normality_stat,
            'normality_p_value': normality_p,
            'is_normal': normality_p > 0.05
        })
        
        # Detección de outliers (método IQR)
        q1, q3 = stats_dict['q25'], stats_dict['q75']
        iqr = stats_dict['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        stats_dict.update({
            'outliers_count': len(outliers),
            'outliers_pct': (len(outliers) / len(data)) * 100,
            'outliers_lower_bound': lower_bound,
            'outliers_upper_bound': upper_bound
        })
        
        return stats_dict
    
    def analyze_categorical_variable(self, column: str) -> Dict:
        """
        Análisis completo de variable categórica
        
        Args:
            column: Nombre de la columna a analizar
            
        Returns:
            Diccionario con estadísticas
        """
        if column not in self.categorical_columns:
            raise ValueError(f"'{column}' no es una variable categórica")
        
        data = self.data[column].dropna()
        value_counts = data.value_counts()
        
        # Estadísticas descriptivas
        stats_dict = {
            'count': len(data),
            'missing': self.data[column].isnull().sum(),
            'missing_pct': (self.data[column].isnull().sum() / len(self.data)) * 100,
            'unique_values': data.nunique(),
            'mode': data.mode().iloc[0] if len(data.mode()) > 0 else None,
            'mode_frequency': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'mode_percentage': (value_counts.iloc[0] / len(data)) * 100 if len(value_counts) > 0 else 0,
            'entropy': stats.entropy(value_counts.values),
            'value_counts': value_counts.to_dict(),
            'frequencies': (value_counts / len(data) * 100).to_dict()
        }
        
        return stats_dict
    
    def create_numeric_visualizations(self, column: str) -> Dict:
        """
        Crea visualizaciones para variable numérica
        
        Args:
            column: Nombre de la columna
            
        Returns:
            Diccionario con figuras de Plotly
        """
        data = self.data[column].dropna()
        figures = {}
        
        # 1. Histograma con curva de densidad
        fig_hist = go.Figure()
        
        # Histograma
        fig_hist.add_trace(go.Histogram(
            x=data,
            nbinsx=50,
            name='Frecuencia',
            opacity=0.7,
            marker_color='#3498db',
            yaxis='y'
        ))
        
        # Curva de densidad (KDE)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            kde_values = kde(x_range)
            
            # Escalar KDE para que se vea bien con el histograma
            kde_scaled = kde_values * len(data) * (data.max() - data.min()) / 50
            
            fig_hist.add_trace(go.Scatter(
                x=x_range,
                y=kde_scaled,
                mode='lines',
                name='Densidad (KDE)',
                line=dict(color='#e74c3c', width=3),
                yaxis='y'
            ))
        except:
            pass
        
        fig_hist.update_layout(
            title=f"Distribución de {column}",
            xaxis_title=column,
            yaxis_title="Frecuencia",
            template="plotly_white",
            height=400
        )
        
        figures['histogram'] = fig_hist
        
        # 2. Boxplot
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=data,
            name=column,
            boxpoints='outliers',
            marker_color='#2ecc71',
            line_color='#27ae60'
        ))
        
        fig_box.update_layout(
            title=f"Boxplot de {column}",
            yaxis_title=column,
            template="plotly_white",
            height=400
        )
        
        figures['boxplot'] = fig_box
        
        # 3. Q-Q Plot para normalidad
        fig_qq = go.Figure()
        
        # Calcular quantiles teóricos y empíricos
        sorted_data = np.sort(data)
        n = len(sorted_data)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
        
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            name='Datos',
            marker=dict(color='#9b59b6', size=4)
        ))
        
        # Línea de referencia
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        
        fig_qq.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Línea Normal',
            line=dict(color='#e74c3c', dash='dash')
        ))
        
        fig_qq.update_layout(
            title=f"Q-Q Plot - {column}",
            xaxis_title="Quantiles Teóricos (Normal)",
            yaxis_title="Quantiles Empíricos",
            template="plotly_white",
            height=400
        )
        
        figures['qqplot'] = fig_qq
        
        # 4. ECDF (Empirical Cumulative Distribution Function)
        fig_ecdf = go.Figure()
        
        sorted_data = np.sort(data)
        y_ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig_ecdf.add_trace(go.Scatter(
            x=sorted_data,
            y=y_ecdf,
            mode='lines',
            name='ECDF',
            line=dict(color='#f39c12', width=2)
        ))
        
        fig_ecdf.update_layout(
            title=f"Función de Distribución Empírica - {column}",
            xaxis_title=column,
            yaxis_title="Probabilidad Acumulada",
            template="plotly_white",
            height=400
        )
        
        figures['ecdf'] = fig_ecdf
        
        return figures
    
    def create_categorical_visualizations(self, column: str) -> Dict:
        """
        Crea visualizaciones para variable categórica
        
        Args:
            column: Nombre de la columna
            
        Returns:
            Diccionario con figuras de Plotly
        """
        data = self.data[column].dropna()
        value_counts = data.value_counts()
        figures = {}
        
        # 1. Gráfico de barras
        fig_bar = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Frecuencias de {column}",
            labels={'x': column, 'y': 'Frecuencia'},
            color=value_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig_bar.update_layout(
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        figures['barplot'] = fig_bar
        
        # 2. Gráfico de torta
        fig_pie = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribución de {column}"
        )
        
        fig_pie.update_layout(
            template="plotly_white",
            height=400
        )
        
        figures['pieplot'] = fig_pie
        
        return figures

def render_univariate_analysis():
    """Renderiza el módulo de análisis univariado en Streamlit"""
    st.title("📈 Análisis Descriptivo Univariado")
    st.markdown("### *Análisis estadístico detallado por variable individual*")
    
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
    analyzer = UnivariateAnalyzer(df)
    
    # Selector de variable
    st.subheader("🎯 Selección de Variable")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Combinar todas las columnas
        all_columns = analyzer.numeric_columns + analyzer.categorical_columns
        
        selected_variable = st.selectbox(
            "Selecciona una variable para analizar:",
            options=all_columns,
            help="Elige la variable que deseas analizar en detalle"
        )
    
    with col2:
        variable_type = "Numérica" if selected_variable in analyzer.numeric_columns else "Categórica"
        st.metric("Tipo de Variable", variable_type)
        
        if variable_type == "Numérica":
            st.metric("Valores Únicos", df[selected_variable].nunique())
        else:
            st.metric("Categorías", df[selected_variable].nunique())
    
    if not selected_variable:
        st.warning("⚠️ Selecciona una variable para continuar.")
        return
    
    # Análisis según el tipo de variable
    if variable_type == "Numérica":
        render_numeric_analysis(analyzer, selected_variable, df)
    else:
        render_categorical_analysis(analyzer, selected_variable, df)

def render_numeric_analysis(analyzer: UnivariateAnalyzer, column: str, df: pd.DataFrame):
    """Renderiza análisis de variable numérica"""
    
    # Análisis estadístico
    stats_result = analyzer.analyze_numeric_variable(column)
    
    # Crear visualizaciones
    figures = analyzer.create_numeric_visualizations(column)
    
    # Mostrar en tabs
    tab1, tab2 = st.tabs(["📊 Estadísticas", "📈 Visualizaciones"])
    
    with tab1:
        st.subheader(f"📊 Estadísticas Descriptivas - {column}")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Media", f"{stats_result['mean']:,.2f}")
            st.metric("Mediana", f"{stats_result['median']:,.2f}")
        
        with col2:
            st.metric("Desv. Estándar", f"{stats_result['std']:,.2f}")
            st.metric("Coef. Variación", f"{stats_result['cv']:.2f}%")
        
        with col3:
            st.metric("Mínimo", f"{stats_result['min']:,.2f}")
            st.metric("Máximo", f"{stats_result['max']:,.2f}")
        
        with col4:
            st.metric("Asimetría", f"{stats_result['skewness']:.3f}")
            st.metric("Curtosis", f"{stats_result['kurtosis']:.3f}")
        
        # Tabla detallada
        st.subheader("📋 Estadísticas Completas")
        
        stats_df = pd.DataFrame([
            ["Observaciones", f"{stats_result['count']:,}"],
            ["Valores Faltantes", f"{stats_result['missing']:,} ({stats_result['missing_pct']:.1f}%)"],
            ["Media", f"{stats_result['mean']:,.4f}"],
            ["Mediana", f"{stats_result['median']:,.4f}"],
            ["Moda", f"{stats_result['mode']:,.4f}"],
            ["Desviación Estándar", f"{stats_result['std']:,.4f}"],
            ["Varianza", f"{stats_result['variance']:,.4f}"],
            ["Rango", f"{stats_result['range']:,.4f}"],
            ["Rango Intercuartílico (IQR)", f"{stats_result['iqr']:,.4f}"],
            ["Percentil 5", f"{stats_result['p5']:,.4f}"],
            ["Cuartil 1 (Q1)", f"{stats_result['q25']:,.4f}"],
            ["Cuartil 3 (Q3)", f"{stats_result['q75']:,.4f}"],
            ["Percentil 95", f"{stats_result['p95']:,.4f}"],
            ["Coeficiente de Variación", f"{stats_result['cv']:.2f}%"],
            ["Asimetría (Skewness)", f"{stats_result['skewness']:.4f}"],
            ["Curtosis", f"{stats_result['kurtosis']:.4f}"]
        ], columns=["Estadística", "Valor"])
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Test de normalidad
        st.subheader("🔍 Test de Normalidad")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Test", stats_result['normality_test'])
            st.metric("Estadístico", f"{stats_result['normality_statistic']:.6f}")
        
        with col2:
            st.metric("P-valor", f"{stats_result['normality_p_value']:.6f}")
            
            if stats_result['is_normal']:
                st.success("✅ Distribución Normal (p > 0.05)")
            else:
                st.warning("⚠️ No sigue distribución normal (p ≤ 0.05)")
        
        # Outliers
        st.subheader("🎯 Detección de Outliers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Outliers", f"{stats_result['outliers_count']:,}")
        
        with col2:
            st.metric("Porcentaje", f"{stats_result['outliers_pct']:.2f}%")
        
        with col3:
            if stats_result['outliers_pct'] < 5:
                st.success("✅ Pocos outliers")
            elif stats_result['outliers_pct'] < 10:
                st.warning("⚠️ Outliers moderados")
            else:
                st.error("❌ Muchos outliers")
        
        st.markdown(f"""
        **Rango normal (método IQR):**
        - Límite inferior: {stats_result['outliers_lower_bound']:,.2f}
        - Límite superior: {stats_result['outliers_upper_bound']:,.2f}
        """)
    
    with tab2:
        st.subheader(f"📈 Visualizaciones - {column}")
        
        # Mostrar gráficos en sub-tabs
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "📊 Histograma", 
            "📦 Boxplot", 
            "📈 Q-Q Plot", 
            "📉 ECDF"
        ])
        
        with subtab1:
            if 'histogram' in figures:
                st.plotly_chart(figures['histogram'], use_container_width=True)
                st.markdown("""
                **Interpretación del Histograma:**
                - Muestra la distribución de frecuencias
                - La curva roja (KDE) suaviza la distribución
                - Identifica modas, asimetría y forma de la distribución
                """)
        
        with subtab2:
            if 'boxplot' in figures:
                st.plotly_chart(figures['boxplot'], use_container_width=True)
                st.markdown("""
                **Interpretación del Boxplot:**
                - La caja muestra Q1, mediana y Q3
                - Los bigotes se extienden hasta 1.5×IQR
                - Los puntos son outliers potenciales
                """)
        
        with subtab3:
            if 'qqplot' in figures:
                st.plotly_chart(figures['qqplot'], use_container_width=True)
                st.markdown("""
                **Interpretación del Q-Q Plot:**
                - Compara quantiles empíricos vs teóricos (normal)
                - Puntos sobre la línea roja = distribución normal
                - Desviaciones indican no-normalidad
                """)
        
        with subtab4:
            if 'ecdf' in figures:
                st.plotly_chart(figures['ecdf'], use_container_width=True)
                st.markdown("""
                **Interpretación del ECDF:**
                - Función de distribución empírica acumulada
                - Muestra la probabilidad de valores ≤ x
                - Útil para comparar distribuciones
                """)

def render_categorical_analysis(analyzer: UnivariateAnalyzer, column: str, df: pd.DataFrame):
    """Renderiza análisis de variable categórica"""
    
    # Análisis estadístico
    stats_result = analyzer.analyze_categorical_variable(column)
    
    # Crear visualizaciones
    figures = analyzer.create_categorical_visualizations(column)
    
    # Mostrar en tabs
    tab1, tab2 = st.tabs(["📊 Estadísticas", "📈 Visualizaciones"])
    
    with tab1:
        st.subheader(f"📊 Estadísticas Descriptivas - {column}")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Observaciones", f"{stats_result['count']:,}")
        
        with col2:
            st.metric("Valores Únicos", stats_result['unique_values'])
        
        with col3:
            st.metric("Moda", str(stats_result['mode']))
        
        with col4:
            st.metric("Entropía", f"{stats_result['entropy']:.3f}")
        
        # Tabla de frecuencias
        st.subheader("📋 Tabla de Frecuencias")
        
        freq_df = pd.DataFrame([
            [cat, freq, f"{(freq/stats_result['count']*100):.2f}%"]
            for cat, freq in stats_result['value_counts'].items()
        ], columns=["Categoría", "Frecuencia", "Porcentaje"])
        
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
        
        # Información adicional
        if stats_result['missing'] > 0:
            st.warning(f"⚠️ Valores faltantes: {stats_result['missing']:,} ({stats_result['missing_pct']:.1f}%)")
        
        st.info(f"💡 La categoría más frecuente es '{stats_result['mode']}' con {stats_result['mode_percentage']:.1f}% de los casos.")
    
    with tab2:
        st.subheader(f"📈 Visualizaciones - {column}")
        
        # Mostrar gráficos en columnas
        col1, col2 = st.columns(2)
        
        with col1:
            if 'barplot' in figures:
                st.plotly_chart(figures['barplot'], use_container_width=True)
        
        with col2:
            if 'pieplot' in figures:
                st.plotly_chart(figures['pieplot'], use_container_width=True)

# ============================================================================
# FUNCIÓN PRINCIPAL PARA INTEGRAR EN APP.PY
# ============================================================================

def render_univariate_module():
    """Función principal para renderizar el módulo univariado"""
    render_univariate_analysis()

if __name__ == "__main__":
    # Para testing
    print("Módulo de análisis univariado cargado correctamente")