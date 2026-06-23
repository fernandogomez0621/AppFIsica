"""
============================================================================
MÓDULO DE RE-ENTRENAMIENTO
============================================================================

Sistema de re-entrenamiento automático de modelos con detección de data drift
y versionado de modelos.

Autor: Sistema de Física
Versión: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats

warnings.filterwarnings('ignore')


class ModelRetrainer:
    """Sistema de re-entrenamiento de modelos"""
    
    def __init__(self):
        """Inicializa el sistema de re-entrenamiento"""
        self.models_dir = Path("models/supervised")
        self.versions_dir = Path("models/versions")
        self.versions_dir.mkdir(parents=True, exist_ok=True)
    
    def get_available_models(self) -> List[str]:
        """Obtiene lista de modelos disponibles"""
        if not self.models_dir.exists():
            return []
        
        model_files = [f.stem.replace('_model', '') for f in self.models_dir.glob('*_model.pkl')]
        return model_files
    
    def load_model_info(self, model_key: str) -> Dict:
        """Carga información del modelo"""
        model_path = self.models_dir / f"{model_key}_model.pkl"
        metrics_path = self.models_dir / f"{model_key}_metrics.json"
        
        info = {}
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                info['model_data'] = model_data
                info['timestamp'] = model_data.get('timestamp', 'Desconocido')
                info['feature_names'] = model_data.get('feature_names', [])
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                info['metrics'] = json.load(f)
        
        return info
    
    def detect_data_drift(self, original_data: pd.DataFrame, new_data: pd.DataFrame,
                         feature_names: List[str]) -> Dict:
        """
        Detecta data drift entre datasets usando tests estadísticos
        
        Args:
            original_data: Dataset original
            new_data: Dataset nuevo
            feature_names: Características a comparar
            
        Returns:
            Resultados del análisis de drift
        """
        drift_results = {
            'features_with_drift': [],
            'drift_scores': {},
            'statistical_tests': {}
        }
        
        for feature in feature_names:
            if feature not in original_data.columns or feature not in new_data.columns:
                continue
            
            # Test de Kolmogorov-Smirnov
            ks_stat, ks_pvalue = stats.ks_2samp(
                original_data[feature].dropna(),
                new_data[feature].dropna()
            )
            
            # Test de Mann-Whitney U (para distribuciones)
            try:
                mw_stat, mw_pvalue = stats.mannwhitneyu(
                    original_data[feature].dropna(),
                    new_data[feature].dropna()
                )
            except:
                mw_pvalue = 1.0
            
            # Calcular diferencia en medias y desviaciones
            mean_diff = abs(new_data[feature].mean() - original_data[feature].mean())
            std_diff = abs(new_data[feature].std() - original_data[feature].std())
            
            # Determinar si hay drift (p-value < 0.05)
            has_drift = ks_pvalue < 0.05 or mw_pvalue < 0.05
            
            drift_results['drift_scores'][feature] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'mw_pvalue': float(mw_pvalue),
                'mean_diff': float(mean_diff),
                'std_diff': float(std_diff),
                'has_drift': has_drift
            }
            
            if has_drift:
                drift_results['features_with_drift'].append(feature)
        
        return drift_results
    
    def create_version_backup(self, model_key: str) -> str:
        """Crea backup versionado del modelo actual"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"{model_key}_v{timestamp}"
        
        # Copiar modelo
        model_src = self.models_dir / f"{model_key}_model.pkl"
        model_dst = self.versions_dir / f"{version_name}_model.pkl"
        
        if model_src.exists():
            import shutil
            shutil.copy2(model_src, model_dst)
        
        # Copiar métricas
        metrics_src = self.models_dir / f"{model_key}_metrics.json"
        metrics_dst = self.versions_dir / f"{version_name}_metrics.json"
        
        if metrics_src.exists():
            import shutil
            shutil.copy2(metrics_src, metrics_dst)
        
        return version_name
    
    def retrain_model(self, model_key: str, new_data: pd.DataFrame,
                     target_col: str = 'nivel_riesgo') -> Dict:
        """
        Re-entrena un modelo con nuevos datos
        
        Args:
            model_key: Clave del modelo a re-entrenar
            new_data: Nuevos datos para entrenamiento
            target_col: Variable objetivo
            
        Returns:
            Resultados del re-entrenamiento
        """
        from src.supervised_models import SupervisedModelTrainer
        
        # Crear entrenador
        trainer = SupervisedModelTrainer()
        
        # Preparar datos
        if not trainer.prepare_data(new_data, target_col):
            raise ValueError("Error preparando datos")
        
        # Re-entrenar modelo
        results = trainer.train_model(model_key, use_grid_search=True)
        
        return results


def render_retraining_module():
    """Renderiza el módulo de re-entrenamiento"""
    st.title("🔄 Re-entrenamiento de Modelos")
    st.markdown("### *Actualización automática con detección de data drift*")
    
    # Crear retrainer
    retrainer = ModelRetrainer()
    
    # Verificar modelos disponibles
    available_models = retrainer.get_available_models()
    
    if not available_models:
        st.error("❌ No hay modelos entrenados. Ve a 'Modelos Supervisados' primero.")
        return
    
    # Selección de modelo
    st.subheader("🤖 Selección de Modelo")
    
    selected_model = st.selectbox(
        "Selecciona el modelo a re-entrenar:",
        options=available_models,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if selected_model:
        # Cargar información del modelo
        model_info = retrainer.load_model_info(selected_model)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modelo", selected_model.replace('_', ' ').title())
        
        with col2:
            if 'metrics' in model_info:
                accuracy = model_info['metrics'].get('accuracy', 0)
                st.metric("Accuracy Actual", f"{accuracy:.3f}")
        
        with col3:
            timestamp = model_info.get('timestamp', 'Desconocido')
            if timestamp != 'Desconocido':
                try:
                    dt = datetime.fromisoformat(timestamp)
                    st.metric("Última Actualización", dt.strftime("%Y-%m-%d"))
                except:
                    st.metric("Última Actualización", "Desconocido")
    
    st.divider()
    
    # Carga de nuevos datos
    st.subheader("📊 Nuevos Datos para Re-entrenamiento")
    
    tab1, tab2 = st.tabs(["📁 Cargar Archivo", "🔍 Detectar Drift"])
    
    with tab1:
        st.markdown("**Opciones de datos:**")
        
        data_option = st.radio(
            "Selecciona la fuente de datos:",
            options=[
                "Usar datos actuales (mismo dataset)",
                "Cargar archivo nuevo (CSV/Excel)"
            ]
        )
        
        new_df = None
        
        if data_option == "Usar datos actuales (mismo dataset)":
            # Usar datos existentes
            data_paths = [
                "data/processed/datos_con_rbm.csv",
                "data/processed/datos_con_caracteristicas.csv",
                "data/processed/datos_credito_hipotecario_realista.csv"
            ]
            
            available_datasets = [p for p in data_paths if os.path.exists(p)]
            
            if available_datasets:
                selected_dataset = st.selectbox(
                    "Dataset:",
                    options=available_datasets,
                    format_func=lambda x: x.split('/')[-1]
                )
                
                new_df = pd.read_csv(selected_dataset)
                st.success(f"✅ Dataset cargado: {len(new_df):,} registros")
            else:
                st.error("❌ No hay datasets disponibles")
        
        else:
            # Cargar archivo nuevo
            uploaded_file = st.file_uploader(
                "Sube un archivo CSV o Excel:",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        new_df = pd.read_csv(uploaded_file)
                    else:
                        new_df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ Archivo cargado: {len(new_df):,} registros, {len(new_df.columns)} variables")
                    
                    # Mostrar muestra
                    with st.expander("Ver muestra de datos"):
                        st.dataframe(new_df.head(), use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error cargando archivo: {e}")
    
    with tab2:
        if new_df is not None and 'feature_names' in model_info:
            st.markdown("**Análisis de Data Drift:**")
            
            # Cargar datos originales
            original_data_path = "data/processed/datos_credito_hipotecario_realista.csv"
            if os.path.exists(original_data_path):
                original_df = pd.read_csv(original_data_path)
                
                if st.button("🔍 Detectar Data Drift", type="primary"):
                    with st.spinner("🔍 Analizando diferencias en distribuciones..."):
                        drift_results = retrainer.detect_data_drift(
                            original_df,
                            new_df,
                            model_info['feature_names']
                        )
                        
                        # Mostrar resultados
                        st.subheader("📊 Resultados del Análisis de Drift")
                        
                        n_features_with_drift = len(drift_results['features_with_drift'])
                        total_features = len(model_info['feature_names'])
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Características Analizadas", total_features)
                        
                        with col2:
                            st.metric("Con Drift Detectado", n_features_with_drift)
                        
                        with col3:
                            drift_percentage = (n_features_with_drift / total_features) * 100
                            st.metric("% con Drift", f"{drift_percentage:.1f}%")
                        
                        # Tabla de características con drift
                        if drift_results['features_with_drift']:
                            st.warning(f"⚠️ Se detectó drift en {n_features_with_drift} características")
                            
                            drift_data = []
                            for feature in drift_results['features_with_drift']:
                                scores = drift_results['drift_scores'][feature]
                                drift_data.append({
                                    'Característica': feature,
                                    'KS p-value': f"{scores['ks_pvalue']:.4f}",
                                    'Diff Media': f"{scores['mean_diff']:.4f}",
                                    'Diff Std': f"{scores['std_diff']:.4f}"
                                })
                            
                            drift_df = pd.DataFrame(drift_data)
                            st.dataframe(drift_df, use_container_width=True, hide_index=True)
                            
                            st.info("💡 **Recomendación:** Re-entrenar el modelo con los nuevos datos")
                        else:
                            st.success("✅ No se detectó drift significativo en las características")
                            st.info("ℹ️ El modelo actual debería seguir funcionando bien")
            else:
                st.warning("⚠️ No se encontraron datos originales para comparar")
        else:
            st.info("ℹ️ Carga datos nuevos primero para analizar drift")
    
    st.divider()
    
    # Re-entrenamiento
    if new_df is not None:
        st.subheader("🚀 Re-entrenamiento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_backup = st.checkbox(
                "Crear backup del modelo actual",
                value=True,
                help="Guarda una versión del modelo antes de re-entrenar"
            )
        
        with col2:
            use_grid_search = st.checkbox(
                "Optimizar hiperparámetros",
                value=True,
                help="Buscar mejores hiperparámetros (más lento)"
            )
        
        if st.button("🔄 RE-ENTRENAR MODELO", type="primary", use_container_width=True):
            with st.spinner("🔄 Re-entrenando modelo..."):
                try:
                    # Crear backup si se solicita
                    if create_backup:
                        version_name = retrainer.create_version_backup(selected_model)
                        st.info(f"💾 Backup creado: {version_name}")
                    
                    # Re-entrenar
                    results = retrainer.retrain_model(selected_model, new_df)
                    
                    st.success("✅ Modelo re-entrenado exitosamente!")
                    
                    # Comparar métricas
                    st.subheader("📊 Comparación de Rendimiento")
                    
                    old_metrics = model_info.get('metrics', {})
                    new_metrics = results['metrics']
                    
                    comparison_data = []
                    metrics_to_compare = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
                    
                    for metric in metrics_to_compare:
                        old_value = old_metrics.get(metric, 0)
                        new_value = new_metrics.get(metric, 0)
                        diff = new_value - old_value
                        
                        comparison_data.append({
                            'Métrica': metric.replace('_', ' ').title(),
                            'Modelo Anterior': f"{old_value:.4f}",
                            'Modelo Nuevo': f"{new_value:.4f}",
                            'Diferencia': f"{diff:+.4f}",
                            'Mejora': '✅' if diff > 0 else ('⚠️' if diff < -0.01 else '➖')
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Gráfico de comparación
                    fig_comparison = go.Figure()
                    
                    metrics_names = [d['Métrica'] for d in comparison_data]
                    old_values = [float(d['Modelo Anterior']) for d in comparison_data]
                    new_values = [float(d['Modelo Nuevo']) for d in comparison_data]
                    
                    fig_comparison.add_trace(go.Bar(
                        name='Modelo Anterior',
                        x=metrics_names,
                        y=old_values,
                        marker_color='#95a5a6'
                    ))
                    
                    fig_comparison.add_trace(go.Bar(
                        name='Modelo Nuevo',
                        x=metrics_names,
                        y=new_values,
                        marker_color='#3498db'
                    ))
                    
                    fig_comparison.update_layout(
                        title="Comparación: Modelo Anterior vs Nuevo",
                        xaxis_title="Métricas",
                        yaxis_title="Valor",
                        barmode='group',
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Recomendación
                    avg_improvement = np.mean([float(d['Diferencia']) for d in comparison_data])
                    
                    if avg_improvement > 0.01:
                        st.success("✅ **Recomendación:** El nuevo modelo es significativamente mejor. Se recomienda usarlo.")
                    elif avg_improvement < -0.01:
                        st.error("❌ **Recomendación:** El nuevo modelo es peor. Considera usar el backup.")
                    else:
                        st.info("ℹ️ **Recomendación:** Rendimiento similar. Puedes usar cualquiera.")
                
                except Exception as e:
                    st.error(f"❌ Error durante re-entrenamiento: {e}")
                    st.exception(e)
    
    # Gestión de versiones
    st.divider()
    st.subheader("📦 Versiones de Modelos")
    
    version_files = list(retrainer.versions_dir.glob("*_model.pkl"))
    
    if version_files:
        st.success(f"✅ {len(version_files)} versiones guardadas")
        
        with st.expander("Ver versiones disponibles"):
            for version_file in sorted(version_files, reverse=True):
                version_name = version_file.stem.replace('_model', '')
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.text(f"📦 {version_name}")
                
                with col2:
                    if st.button("🔙 Restaurar", key=f"restore_{version_name}"):
                        try:
                            import shutil
                            # Restaurar modelo
                            shutil.copy2(
                                version_file,
                                retrainer.models_dir / f"{selected_model}_model.pkl"
                            )
                            
                            # Restaurar métricas si existen
                            metrics_file = version_file.parent / f"{version_name}_metrics.json"
                            if metrics_file.exists():
                                shutil.copy2(
                                    metrics_file,
                                    retrainer.models_dir / f"{selected_model}_metrics.json"
                                )
                            
                            st.success(f"✅ Modelo restaurado desde: {version_name}")
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ Error restaurando: {e}")
    else:
        st.info("ℹ️ No hay versiones guardadas aún")


if __name__ == "__main__":
    print("Módulo de re-entrenamiento cargado correctamente")