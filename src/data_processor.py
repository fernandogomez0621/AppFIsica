"""
============================================================================
MÓDULO DE PROCESAMIENTO DE DATOS
============================================================================

Carga, validación y procesamiento de datasets de crédito hipotecario.
Incluye validaciones automáticas, limpieza de datos y reportes de calidad.

Autor: Sistema de Física
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Tuple, Optional
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class DataProcessor:
    """Procesador de datos de crédito hipotecario"""
    
    def __init__(self):
        """Inicializa el procesador"""
        self.required_columns = [
            'edad', 'salario_mensual', 'puntaje_datacredito', 
            'valor_inmueble', 'monto_credito', 'nivel_riesgo'
        ]
        
        self.numeric_ranges = {
            'edad': (18, 80),
            'salario_mensual': (1000000, 50000000),
            'puntaje_datacredito': (150, 950),
            'valor_inmueble': (20000000, 2000000000),
            'monto_credito': (10000000, 1500000000),
            'tasa_interes_anual': (5.0, 25.0),
            'plazo_credito': (5, 30),
            'dti': (0, 60),
            'ltv': (0, 100)
        }
        
        self.categorical_columns = [
            'ciudad', 'nivel_educacion', 'tipo_empleo', 
            'estado_civil', 'nivel_riesgo'
        ]
        
        self.validation_report = {}
        self.processing_report = {}
    
    def load_data(self, file_path: str = None, uploaded_file = None) -> pd.DataFrame:
        """
        Carga datos desde archivo o upload de Streamlit
        
        Args:
            file_path: Ruta al archivo
            uploaded_file: Archivo subido en Streamlit
            
        Returns:
            DataFrame cargado
        """
        try:
            if uploaded_file is not None:
                # Detectar tipo de archivo
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                elif file_extension == 'parquet':
                    df = pd.read_parquet(uploaded_file)
                else:
                    raise ValueError(f"Formato no soportado: {file_extension}")
                
                st.success(f"✅ Archivo cargado: {uploaded_file.name}")
                
            elif file_path:
                file_extension = file_path.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df = pd.read_csv(file_path)
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(file_path)
                elif file_extension == 'parquet':
                    df = pd.read_parquet(file_path)
                else:
                    raise ValueError(f"Formato no soportado: {file_extension}")
                
                st.success(f"✅ Archivo cargado: {file_path}")
            
            else:
                raise ValueError("Debe proporcionar file_path o uploaded_file")
            
            st.info(f"📊 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error cargando archivo: {e}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Valida la calidad y consistencia de los datos
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Reporte de validación
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # 1. Verificar columnas requeridas
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            report['errors'].append(f"Columnas faltantes: {missing_cols}")
        else:
            report['suggestions'].append("✅ Todas las columnas requeridas están presentes")
        
        # 2. Verificar valores faltantes
        missing_summary = df.isnull().sum()
        missing_pct = (missing_summary / len(df)) * 100
        
        high_missing = missing_pct[missing_pct > 20].index.tolist()
        if high_missing:
            report['warnings'].append(f"Columnas con >20% valores faltantes: {high_missing}")
        
        moderate_missing = missing_pct[(missing_pct > 5) & (missing_pct <= 20)].index.tolist()
        if moderate_missing:
            report['warnings'].append(f"Columnas con 5-20% valores faltantes: {moderate_missing}")
        
        # 3. Verificar rangos lógicos
        for col, (min_val, max_val) in self.numeric_ranges.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)][col].count()
                if out_of_range > 0:
                    report['warnings'].append(
                        f"{col}: {out_of_range} valores fuera del rango [{min_val}, {max_val}]"
                    )
        
        # 4. Verificar inconsistencias lógicas
        if 'monto_credito' in df.columns and 'valor_inmueble' in df.columns:
            inconsistent = (df['monto_credito'] > df['valor_inmueble']).sum()
            if inconsistent > 0:
                report['errors'].append(f"Inconsistencia: {inconsistent} casos con monto_credito > valor_inmueble")
        
        if 'salario_mensual' in df.columns and 'egresos_mensuales' in df.columns:
            inconsistent = (df['egresos_mensuales'] > df['salario_mensual']).sum()
            if inconsistent > 0:
                report['warnings'].append(f"Advertencia: {inconsistent} casos con egresos > salario")
        
        # 5. Detectar duplicados
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report['warnings'].append(f"Filas duplicadas: {duplicates}")
        
        # 6. Detectar outliers extremos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Outliers extremos (3 * IQR)
                extreme_outliers = df[
                    (df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)
                ][col].count()
                
                if extreme_outliers > len(df) * 0.05:  # Más del 5%
                    report['warnings'].append(f"{col}: {extreme_outliers} outliers extremos ({extreme_outliers/len(df)*100:.1f}%)")
        
        self.validation_report = report
        return report
    
    def clean_data(self, df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   handle_missing: str = 'impute',
                   handle_outliers: str = 'keep',
                   normalize_numeric: bool = False) -> pd.DataFrame:
        """
        Limpia y procesa los datos
        
        Args:
            df: DataFrame a procesar
            remove_duplicates: Si eliminar duplicados
            handle_missing: 'drop', 'impute', 'keep'
            handle_outliers: 'remove', 'cap', 'keep'
            normalize_numeric: Si normalizar variables numéricas
            
        Returns:
            DataFrame procesado
        """
        df_clean = df.copy()
        processing_steps = []
        
        # 1. Eliminar duplicados
        if remove_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_dups = initial_rows - len(df_clean)
            if removed_dups > 0:
                processing_steps.append(f"✅ Eliminados {removed_dups} duplicados")
        
        # 2. Manejar valores faltantes
        if handle_missing == 'drop':
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            removed_na = initial_rows - len(df_clean)
            if removed_na > 0:
                processing_steps.append(f"✅ Eliminadas {removed_na} filas con valores faltantes")
        
        elif handle_missing == 'impute':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
            
            # Imputar numéricas con mediana
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    processing_steps.append(f"✅ {col}: imputado con mediana ({median_val:.2f})")
            
            # Imputar categóricas con moda
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
                    processing_steps.append(f"✅ {col}: imputado con moda ({mode_val})")
        
        # 3. Manejar outliers
        if handle_outliers in ['remove', 'cap']:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    if handle_outliers == 'remove':
                        df_clean = df_clean[~outliers_mask]
                        processing_steps.append(f"✅ {col}: eliminados {outliers_count} outliers")
                    
                    elif handle_outliers == 'cap':
                        df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                        df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                        processing_steps.append(f"✅ {col}: {outliers_count} outliers limitados")
        
        # 4. Normalizar variables numéricas
        if normalize_numeric:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            
            df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
            processing_steps.append(f"✅ Normalizadas {len(numeric_cols)} variables numéricas")
        
        # 5. Codificar variables categóricas
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'nivel_riesgo']  # Preservar target
        
        for col in categorical_cols:
            if df_clean[col].nunique() <= 10:  # One-hot para pocas categorías
                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
                df_clean = pd.concat([df_clean.drop(col, axis=1), dummies], axis=1)
                processing_steps.append(f"✅ {col}: codificado con One-Hot ({dummies.shape[1]} variables)")
            else:  # Label encoding para muchas categorías
                le = LabelEncoder()
                df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
                processing_steps.append(f"✅ {col}: codificado con Label Encoding")
        
        self.processing_report = {
            'timestamp': datetime.now().isoformat(),
            'initial_shape': df.shape,
            'final_shape': df_clean.shape,
            'steps': processing_steps
        }
        
        return df_clean
    
    def create_quality_report_visualizations(self, df: pd.DataFrame) -> Dict:
        """
        Crea visualizaciones del reporte de calidad
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con figuras
        """
        figures = {}
        
        # 1. Heatmap de valores faltantes
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        if missing_data.sum() > 0:
            fig_missing = go.Figure(data=go.Heatmap(
                z=df.isnull().values.T,
                y=df.columns,
                colorscale=[[0, '#2ecc71'], [1, '#e74c3c']],
                showscale=True,
                colorbar=dict(title="Faltante", tickvals=[0, 1], ticktext=['No', 'Sí'])
            ))
            
            fig_missing.update_layout(
                title="Mapa de Valores Faltantes",
                xaxis_title="Registros",
                yaxis_title="Variables",
                height=max(400, len(df.columns) * 20),
                template="plotly_white"
            )
            
            figures['missing_heatmap'] = fig_missing
        
        # 2. Gráfico de barras de valores faltantes
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Variable': missing_data.index,
                'Valores_Faltantes': missing_data.values,
                'Porcentaje': missing_pct.values
            }).sort_values('Valores_Faltantes', ascending=True)
            
            fig_missing_bar = px.bar(
                missing_df,
                x='Valores_Faltantes',
                y='Variable',
                orientation='h',
                title="Valores Faltantes por Variable",
                labels={'Valores_Faltantes': 'Cantidad de Valores Faltantes'},
                color='Porcentaje',
                color_continuous_scale='Reds'
            )
            
            fig_missing_bar.update_layout(
                template="plotly_white",
                height=max(400, len(missing_df) * 25)
            )
            
            figures['missing_barplot'] = fig_missing_bar
        
        # 3. Distribución de tipos de datos
        dtype_counts = df.dtypes.value_counts()
        
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index.astype(str),
            title="Distribución de Tipos de Datos"
        )
        
        fig_dtypes.update_layout(
            template="plotly_white",
            height=400
        )
        
        figures['dtypes_pie'] = fig_dtypes
        
        return figures

def render_data_loader():
    """Renderiza el módulo de carga de datos en Streamlit"""
    st.title("📁 Carga y Validación de Datos")
    st.markdown("### *Carga, valida y procesa datasets de crédito hipotecario*")
    
    # Crear procesador
    processor = DataProcessor()
    
    # Opciones de carga
    st.subheader("📤 Cargar Dataset")
    
    load_option = st.radio(
        "Selecciona el origen de los datos:",
        ["📁 Subir archivo", "💾 Usar datos generados"],
        horizontal=True
    )
    
    df = None
    
    if load_option == "📁 Subir archivo":
        st.markdown("**Formatos soportados:** CSV, Excel (.xlsx, .xls), Parquet")
        
        uploaded_file = st.file_uploader(
            "Arrastra tu archivo aquí o haz clic para seleccionar",
            type=['csv', 'xlsx', 'xls', 'parquet'],
            help="Sube un archivo con datos de crédito hipotecario"
        )
        
        if uploaded_file:
            df = processor.load_data(uploaded_file=uploaded_file)
    
    else:  # Usar datos generados
        if os.path.exists("data/processed/datos_credito_hipotecario_realista.csv"):
            df = processor.load_data(file_path="data/processed/datos_credito_hipotecario_realista.csv")
        else:
            st.warning("⚠️ No hay datos generados. Ve a 'Generar Datos' primero.")
            return
    
    if df is None:
        st.info("👆 Selecciona un archivo para continuar.")
        return
    
    # Mostrar preview de datos
    st.subheader("👀 Vista Previa de los Datos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", len(df.columns))
    with col3:
        st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Mostrar muestra
    st.dataframe(df.head(10), use_container_width=True)
    
    # Validación de datos
    st.divider()
    st.subheader("✅ Validación de Calidad")
    
    if st.button("🔍 Ejecutar Validación", type="primary"):
        with st.spinner("🔍 Validando calidad de datos..."):
            validation_report = processor.validate_data(df)
            
            # Mostrar resultados de validación
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Errores", len(validation_report['errors']))
            with col2:
                st.metric("Advertencias", len(validation_report['warnings']))
            with col3:
                st.metric("Sugerencias", len(validation_report['suggestions']))
            
            # Mostrar detalles
            if validation_report['errors']:
                st.error("❌ **Errores encontrados:**")
                for error in validation_report['errors']:
                    st.error(f"• {error}")
            
            if validation_report['warnings']:
                st.warning("⚠️ **Advertencias:**")
                for warning in validation_report['warnings']:
                    st.warning(f"• {warning}")
            
            if validation_report['suggestions']:
                st.success("✅ **Estado positivo:**")
                for suggestion in validation_report['suggestions']:
                    st.success(f"• {suggestion}")
            
            # Crear visualizaciones de calidad
            figures = processor.create_quality_report_visualizations(df)
            
            if figures:
                st.subheader("📊 Visualizaciones de Calidad")
                
                # Mostrar en tabs
                tabs = []
                if 'missing_heatmap' in figures:
                    tabs.append("🔥 Heatmap Faltantes")
                if 'missing_barplot' in figures:
                    tabs.append("📊 Barras Faltantes")
                if 'dtypes_pie' in figures:
                    tabs.append("🥧 Tipos de Datos")
                
                if tabs:
                    tab_objects = st.tabs(tabs)
                    
                    tab_idx = 0
                    if 'missing_heatmap' in figures:
                        with tab_objects[tab_idx]:
                            st.plotly_chart(figures['missing_heatmap'], use_container_width=True)
                        tab_idx += 1
                    
                    if 'missing_barplot' in figures:
                        with tab_objects[tab_idx]:
                            st.plotly_chart(figures['missing_barplot'], use_container_width=True)
                        tab_idx += 1
                    
                    if 'dtypes_pie' in figures:
                        with tab_objects[tab_idx]:
                            st.plotly_chart(figures['dtypes_pie'], use_container_width=True)
    
    # Procesamiento de datos
    st.divider()
    st.subheader("🔧 Procesamiento de Datos")
    
    with st.expander("⚙️ Configuración de Procesamiento", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            remove_duplicates = st.checkbox("Eliminar duplicados", value=True)
            
            handle_missing = st.selectbox(
                "Manejar valores faltantes:",
                options=['keep', 'impute', 'drop'],
                index=1,
                help="keep=mantener, impute=imputar, drop=eliminar filas"
            )
        
        with col2:
            handle_outliers = st.selectbox(
                "Manejar outliers:",
                options=['keep', 'cap', 'remove'],
                index=0,
                help="keep=mantener, cap=limitar, remove=eliminar"
            )
            
            normalize_numeric = st.checkbox(
                "Normalizar variables numéricas",
                value=False,
                help="Aplicar StandardScaler a variables numéricas"
            )
    
    if st.button("🚀 Procesar Datos", type="primary"):
        with st.spinner("🔧 Procesando datos..."):
            try:
                df_processed = processor.clean_data(
                    df,
                    remove_duplicates=remove_duplicates,
                    handle_missing=handle_missing,
                    handle_outliers=handle_outliers,
                    normalize_numeric=normalize_numeric
                )
                
                # Mostrar resultados del procesamiento
                st.success("✅ Datos procesados exitosamente!")
                
                # Comparación antes/después
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Antes del procesamiento:**")
                    st.metric("Filas", f"{len(df):,}")
                    st.metric("Columnas", len(df.columns))
                    st.metric("Valores Faltantes", df.isnull().sum().sum())
                
                with col2:
                    st.markdown("**Después del procesamiento:**")
                    st.metric("Filas", f"{len(df_processed):,}")
                    st.metric("Columnas", len(df_processed.columns))
                    st.metric("Valores Faltantes", df_processed.isnull().sum().sum())
                
                # Mostrar pasos de procesamiento
                if processor.processing_report['steps']:
                    st.subheader("📋 Pasos de Procesamiento Ejecutados")
                    for step in processor.processing_report['steps']:
                        st.success(step)
                
                # Guardar datos procesados
                os.makedirs("data/processed", exist_ok=True)
                processed_path = "data/processed/datos_procesados.csv"
                df_processed.to_csv(processed_path, index=False)
                
                st.success(f"💾 Datos procesados guardados en: {processed_path}")
                
                # Mostrar muestra de datos procesados
                st.subheader("👀 Vista Previa de Datos Procesados")
                st.dataframe(df_processed.head(10), use_container_width=True)
                
                # Botón de descarga
                csv = df_processed.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Datos Procesados",
                    data=csv,
                    file_name=f"datos_procesados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error procesando datos: {e}")
                st.exception(e)

# ============================================================================
# FUNCIÓN PRINCIPAL PARA INTEGRAR EN APP.PY
# ============================================================================

def render_data_processor_module():
    """Función principal para renderizar el módulo de procesamiento"""
    render_data_loader()

if __name__ == "__main__":
    # Para testing
    print("Módulo de procesamiento de datos cargado correctamente")