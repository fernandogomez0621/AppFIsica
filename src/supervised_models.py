"""
============================================================================
MÓDULO DE MODELOS SUPERVISADOS
============================================================================

Entrenamiento y evaluación de múltiples modelos de clasificación de riesgo
crediticio con integración de características RBM.

Autor: Sistema de Física
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, cohen_kappa_score,
    matthews_corrcoef
)
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

class SupervisedModelTrainer:
    """Entrenador de modelos supervisados para riesgo crediticio"""
    
    def __init__(self):
        """Inicializa el entrenador"""
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.X_holdout = None
        self.y_train = None
        self.y_test = None
        self.y_holdout = None
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Configuración de modelos
        self.model_configs = {
            'logistic': {
                'name': 'Logistic Regression',
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'random_forest': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'xgboost': {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'lightgbm': {
                'name': 'LightGBM',
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'svm': {
                'name': 'Support Vector Machine',
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear']
                }
            },
            'mlp': {
                'name': 'Multi-Layer Perceptron',
                'model': MLPClassifier(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50)],
                    'alpha': [0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'nivel_riesgo',
                    test_size: float = 0.2, holdout_size: float = 0.1) -> bool:
        """
        Prepara datos para entrenamiento
        
        Args:
            df: DataFrame con datos
            target_col: Variable objetivo
            test_size: Proporción para testing
            holdout_size: Proporción para holdout
            
        Returns:
            True si exitoso
        """
        try:
            # Verificar que existe la variable objetivo
            if target_col not in df.columns:
                raise ValueError(f"Variable objetivo '{target_col}' no encontrada")
            
            # Preparar características (solo numéricas)
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remover variables que no deben usarse como features
            exclude_cols = ['rechazo_automatico', 'puntaje_riesgo']
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
            
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df[target_col]
            
            # Codificar variable objetivo
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split estratificado: 70% train, 20% test, 10% holdout
            X_temp, self.X_holdout, y_temp, self.y_holdout = train_test_split(
                X, y_encoded, test_size=holdout_size, random_state=42, stratify=y_encoded
            )
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_temp, y_temp, test_size=test_size/(1-holdout_size), random_state=42, stratify=y_temp
            )
            
            # Escalar características
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            self.X_holdout_scaled = self.scaler.transform(self.X_holdout)
            
            self.feature_names = feature_cols
            
            print(f"✅ Datos preparados:")
            print(f"  - Entrenamiento: {self.X_train.shape[0]} muestras")
            print(f"  - Testing: {self.X_test.shape[0]} muestras")
            print(f"  - Holdout: {self.X_holdout.shape[0]} muestras")
            print(f"  - Características: {len(self.feature_names)}")
            print(f"  - Clases: {len(self.label_encoder.classes_)}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error preparando datos: {e}")
            return False
    
    def train_model(self, model_key: str, use_grid_search: bool = True) -> Dict:
        """
        Entrena un modelo específico
        
        Args:
            model_key: Clave del modelo a entrenar
            use_grid_search: Si usar búsqueda de hiperparámetros
            
        Returns:
            Resultados del entrenamiento
        """
        if model_key not in self.model_configs:
            raise ValueError(f"Modelo '{model_key}' no configurado")
        
        config = self.model_configs[model_key]
        
        print(f"\n🚀 Entrenando {config['name']}...")
        
        if use_grid_search and config['params']:
            # Búsqueda de hiperparámetros
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
        else:
            # Entrenamiento directo
            best_model = config['model']
            best_model.fit(self.X_train_scaled, self.y_train)
            best_params = {}
        
        # Predicciones
        y_train_pred = best_model.predict(self.X_train_scaled)
        y_test_pred = best_model.predict(self.X_test_scaled)
        
        # Probabilidades (si el modelo las soporta)
        try:
            y_test_proba = best_model.predict_proba(self.X_test_scaled)
        except:
            y_test_proba = None
        
        # Calcular métricas
        metrics = self._calculate_metrics(self.y_test, y_test_pred, y_test_proba)
        
        # Guardar modelo
        model_path = f"models/supervised/{model_key}_model.pkl"
        os.makedirs("models/supervised", exist_ok=True)
        
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'best_params': best_params,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Guardar métricas en JSON
        metrics_path = f"models/supervised/{model_key}_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convertir numpy types a tipos nativos de Python
            metrics_serializable = self._make_serializable(metrics)
            json.dump(metrics_serializable, f, indent=2)
        
        results = {
            'model': best_model,
            'best_params': best_params,
            'metrics': metrics,
            'predictions': {
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba
            }
        }
        
        print(f"✅ {config['name']} entrenado")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - F1-Score: {metrics['f1_weighted']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: Optional[np.ndarray] = None) -> Dict:
        """Calcula métricas de evaluación completas"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'matthews_corr': matthews_corrcoef(y_true, y_pred)
        }
        
        # AUC si hay probabilidades
        if y_proba is not None:
            try:
                if len(self.label_encoder.classes_) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except:
                metrics['roc_auc'] = np.nan
        
        # Matriz de confusión
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Reporte de clasificación
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        return metrics
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convierte objetos numpy a tipos serializables"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def train_all_models(self, selected_models: List[str], use_grid_search: bool = True) -> Dict:
        """
        Entrena todos los modelos seleccionados
        
        Args:
            selected_models: Lista de modelos a entrenar
            use_grid_search: Si usar búsqueda de hiperparámetros
            
        Returns:
            Resultados de todos los modelos
        """
        all_results = {}
        
        for model_key in selected_models:
            if model_key in self.model_configs:
                try:
                    results = self.train_model(model_key, use_grid_search)
                    all_results[model_key] = results
                except Exception as e:
                    st.error(f"❌ Error entrenando {self.model_configs[model_key]['name']}: {e}")
        
        return all_results
    
    def create_comparison_visualizations(self, results: Dict) -> Dict:
        """
        Crea visualizaciones comparativas de modelos
        
        Args:
            results: Resultados de múltiples modelos
            
        Returns:
            Diccionario con figuras
        """
        figures = {}
        
        if not results:
            return figures
        
        # 1. Comparación de métricas principales
        model_names = []
        accuracies = []
        f1_scores = []
        roc_aucs = []
        
        for model_key, result in results.items():
            model_names.append(self.model_configs[model_key]['name'])
            accuracies.append(result['metrics']['accuracy'])
            f1_scores.append(result['metrics']['f1_weighted'])
            roc_aucs.append(result['metrics'].get('roc_auc', 0))
        
        # Gráfico de barras comparativo
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Accuracy',
            x=model_names,
            y=accuracies,
            marker_color='#3498db'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='F1-Score',
            x=model_names,
            y=f1_scores,
            marker_color='#e74c3c'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='ROC-AUC',
            x=model_names,
            y=roc_aucs,
            marker_color='#2ecc71'
        ))
        
        fig_comparison.update_layout(
            title="Comparación de Modelos - Métricas Principales",
            xaxis_title="Modelos",
            yaxis_title="Score",
            barmode='group',
            template="plotly_white",
            height=500
        )
        
        figures['model_comparison'] = fig_comparison
        
        # 2. Curvas ROC superpuestas (si hay probabilidades)
        fig_roc = go.Figure()
        
        for model_key, result in results.items():
            if result['predictions']['y_test_proba'] is not None:
                y_test_proba = result['predictions']['y_test_proba']
                
                # Para multiclase, usar One-vs-Rest
                if len(self.label_encoder.classes_) > 2:
                    # Tomar la clase de mayor riesgo (última clase)
                    fpr, tpr, _ = roc_curve(
                        (self.y_test == len(self.label_encoder.classes_) - 1).astype(int),
                        y_test_proba[:, -1]
                    )
                else:
                    fpr, tpr, _ = roc_curve(self.y_test, y_test_proba[:, 1])
                
                auc_score = result['metrics'].get('roc_auc', 0)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f"{self.model_configs[model_key]['name']} (AUC={auc_score:.3f})",
                    line=dict(width=2)
                ))
        
        # Línea diagonal
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title="Curvas ROC - Comparación de Modelos",
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            template="plotly_white",
            height=500
        )
        
        figures['roc_curves'] = fig_roc
        
        return figures
    
    def create_confusion_matrix_plot(self, model_key: str, results: Dict) -> go.Figure:
        """
        Crea visualización de matriz de confusión
        
        Args:
            model_key: Clave del modelo
            results: Resultados del modelo
            
        Returns:
            Figura de Plotly
        """
        cm = np.array(results['metrics']['confusion_matrix'])
        class_names = self.label_encoder.classes_
        
        # Normalizar matriz de confusión
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 14},
            showscale=True,
            colorbar=dict(title="Proporción")
        ))
        
        fig.update_layout(
            title=f"Matriz de Confusión - {self.model_configs[model_key]['name']}",
            xaxis_title="Predicción",
            yaxis_title="Real",
            template="plotly_white",
            height=400
        )
        
        return fig

def render_supervised_models():
    """Renderiza el módulo de modelos supervisados en Streamlit"""
    st.title("🤖 Modelos Supervisados")
    st.markdown("### *Entrenamiento y evaluación de modelos de clasificación*")
    
    # Verificar datos
    data_paths = [
        "data/processed/datos_con_caracteristicas.csv",
        "data/processed/datos_con_rbm.csv",
        "data/processed/datos_credito_hipotecario_realista.csv"
    ]
    
    available_datasets = [path for path in data_paths if os.path.exists(path)]
    
    if not available_datasets:
        st.error("❌ No hay datos disponibles. Ve a 'Generar Datos' primero.")
        return
    
    # Selección de dataset
    st.subheader("📊 Selección de Dataset")
    
    dataset_options = {
        "data/processed/datos_credito_hipotecario_realista.csv": "📊 Datos Originales",
        "data/processed/datos_con_caracteristicas.csv": "🔧 Con Ingeniería de Características",
        "data/processed/datos_con_rbm.csv": "⚡ Con Características RBM"
    }
    
    available_options = {path: name for path, name in dataset_options.items() if path in available_datasets}
    
    selected_dataset = st.selectbox(
        "Selecciona el dataset para entrenamiento:",
        options=list(available_options.keys()),
        format_func=lambda x: available_options[x],
        help="Elige el dataset con las características que deseas usar"
    )
    
    # Cargar datos seleccionados
    @st.cache_data
    def load_selected_data(path):
        return pd.read_csv(path)
    
    df = load_selected_data(selected_dataset)
    st.success(f"✅ Dataset cargado: {len(df):,} registros, {len(df.columns)} variables")
    
    # Crear entrenador
    trainer = SupervisedModelTrainer()
    
    # Configuración de entrenamiento
    st.subheader("⚙️ Configuración de Entrenamiento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Selección de Modelos:**")
        
        selected_models = st.multiselect(
            "Modelos a entrenar:",
            options=list(trainer.model_configs.keys()),
            default=['logistic', 'random_forest', 'xgboost'],
            format_func=lambda x: trainer.model_configs[x]['name']
        )
        
        use_grid_search = st.checkbox(
            "Optimización de hiperparámetros",
            value=True,
            help="Usar GridSearchCV para encontrar mejores parámetros"
        )
    
    with col2:
        st.markdown("**División de Datos:**")
        st.markdown("""
        - **70%** Entrenamiento (con validación cruzada 5-fold)
        - **20%** Testing (evaluación final)
        - **10%** Holdout (simulación de producción)
        """)
        
        target_col = st.selectbox(
            "Variable objetivo:",
            options=['nivel_riesgo'],
            help="Variable a predecir"
        )
    
    # Entrenamiento
    if selected_models and st.button("🚀 Entrenar Modelos", type="primary"):
        with st.spinner("🤖 Entrenando modelos de Machine Learning..."):
            try:
                # Preparar datos
                if trainer.prepare_data(df, target_col):
                    
                    # Entrenar modelos seleccionados
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_results = {}
                    
                    for i, model_key in enumerate(selected_models):
                        status_text.text(f"Entrenando {trainer.model_configs[model_key]['name']}...")
                        
                        results = trainer.train_model(model_key, use_grid_search)
                        all_results[model_key] = results
                        
                        progress_bar.progress((i + 1) / len(selected_models))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Guardar resultados en session state
                    st.session_state.model_results = all_results
                    st.session_state.model_trainer = trainer
                    
                    st.success(f"✅ {len(selected_models)} modelos entrenados exitosamente!")
                    
                    # Mostrar tabla de resultados
                    st.subheader("📊 Resultados de Entrenamiento")
                    
                    results_data = []
                    for model_key, result in all_results.items():
                        results_data.append([
                            trainer.model_configs[model_key]['name'],
                            f"{result['metrics']['accuracy']:.4f}",
                            f"{result['metrics']['f1_weighted']:.4f}",
                            f"{result['metrics']['precision_weighted']:.4f}",
                            f"{result['metrics']['recall_weighted']:.4f}",
                            f"{result['metrics'].get('roc_auc', 0):.4f}"
                        ])
                    
                    results_df = pd.DataFrame(results_data, columns=[
                        'Modelo', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC'
                    ])
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Crear visualizaciones comparativas
                    comparison_figures = trainer.create_comparison_visualizations(all_results)
                    
                    if comparison_figures:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'model_comparison' in comparison_figures:
                                st.plotly_chart(comparison_figures['model_comparison'], use_container_width=True)
                        
                        with col2:
                            if 'roc_curves' in comparison_figures:
                                st.plotly_chart(comparison_figures['roc_curves'], use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {e}")
                st.exception(e)
    
    # Análisis detallado de modelos
    if 'model_results' in st.session_state:
        st.divider()
        st.subheader("🔍 Análisis Detallado de Modelos")
        
        model_results = st.session_state.model_results
        trainer = st.session_state.model_trainer
        
        # Selector de modelo para análisis detallado
        selected_model_key = st.selectbox(
            "Selecciona modelo para análisis detallado:",
            options=list(model_results.keys()),
            format_func=lambda x: trainer.model_configs[x]['name']
        )
        
        if selected_model_key:
            result = model_results[selected_model_key]
            
            # Mostrar métricas detalladas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Métricas de Clasificación:**")
                
                metrics_df = pd.DataFrame([
                    ["Accuracy", f"{result['metrics']['accuracy']:.4f}"],
                    ["F1-Score (Weighted)", f"{result['metrics']['f1_weighted']:.4f}"],
                    ["Precision (Weighted)", f"{result['metrics']['precision_weighted']:.4f}"],
                    ["Recall (Weighted)", f"{result['metrics']['recall_weighted']:.4f}"],
                    ["Cohen's Kappa", f"{result['metrics']['cohen_kappa']:.4f}"],
                    ["Matthews Correlation", f"{result['metrics']['matthews_corr']:.4f}"],
                    ["ROC-AUC", f"{result['metrics'].get('roc_auc', 0):.4f}"]
                ], columns=["Métrica", "Valor"])
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Matriz de confusión
                fig_cm = trainer.create_confusion_matrix_plot(selected_model_key, result)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Importancia de características (si está disponible)
            if hasattr(result['model'], 'feature_importances_'):
                st.subheader("📊 Importancia de Características")
                
                importances = result['model'].feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Característica': trainer.feature_names,
                    'Importancia': importances
                }).sort_values('Importancia', ascending=False)
                
                # Top 15 características
                top_features = feature_importance_df.head(15)
                
                fig_importance = px.bar(
                    top_features,
                    x='Importancia',
                    y='Característica',
                    orientation='h',
                    title=f"Top 15 Características - {trainer.model_configs[selected_model_key]['name']}"
                )
                
                fig_importance.update_layout(
                    template="plotly_white",
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)

def render_supervised_models_module():
    """Función principal para renderizar el módulo de modelos supervisados"""
    render_supervised_models()

if __name__ == "__main__":
    print("Módulo de modelos supervisados cargado correctamente")