"""
============================================================================
MÓDULO DE PREDICCIÓN
============================================================================

Sistema de predicción de riesgo crediticio para nuevos solicitantes
con formulario interactivo y explicaciones detalladas.

Autor: Sistema de Física
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

class CreditRiskPredictor:
    """Predictor de riesgo crediticio"""
    
    def __init__(self):
        """Inicializa el predictor"""
        self.available_models = {}
        self.selected_model = None
        self.model_data = None
        self.feature_engineer = None
        
        # Cargar modelos disponibles
        self._load_available_models()
    
    def _load_available_models(self):
        """Carga lista de modelos disponibles"""
        models_dir = "models/supervised"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
            
            for model_file in model_files:
                model_key = model_file.replace('_model.pkl', '')
                model_path = os.path.join(models_dir, model_file)
                
                # Cargar métricas si existen
                metrics_path = os.path.join(models_dir, f"{model_key}_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
                
                self.available_models[model_key] = {
                    'path': model_path,
                    'metrics': metrics,
                    'name': model_key.replace('_', ' ').title()
                }
    
    def load_model(self, model_key: str) -> bool:
        """
        Carga un modelo entrenado
        
        Args:
            model_key: Clave del modelo a cargar
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            if model_key not in self.available_models:
                return False
            
            model_path = self.available_models[model_key]['path']
            
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.selected_model = model_key
            
            # Cargar feature engineer si existe
            try:
                from src.feature_engineering import FeatureEngineer
                self.feature_engineer = FeatureEngineer
            except:
                self.feature_engineer = None
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error cargando modelo: {e}")
            return False
    
    def create_prediction_form(self) -> Dict:
        """
        Crea formulario interactivo para capturar datos del solicitante
        
        Returns:
            Diccionario con datos del formulario
        """
        st.subheader("📝 Datos del Solicitante")
        
        # Organizar en tabs para mejor UX
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 Personal", 
            "💼 Laboral", 
            "💰 Financiero", 
            "🏠 Inmueble"
        ])
        
        form_data = {}
        
        # ==================== TAB 1: DATOS PERSONALES ====================
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                form_data['edad'] = st.number_input(
                    "Edad:",
                    min_value=18,
                    max_value=80,
                    value=35,
                    help="Edad del solicitante en años"
                )
                
                form_data['estado_civil'] = st.selectbox(
                    "Estado civil:",
                    options=['Soltero', 'Casado', 'Unión Libre', 'Divorciado', 'Viudo']
                )
                
                form_data['nivel_educacion'] = st.selectbox(
                    "Nivel de educación:",
                    options=['Bachiller', 'Técnico', 'Profesional', 'Posgrado']
                )
            
            with col2:
                form_data['ciudad'] = st.selectbox(
                    "Ciudad:",
                    options=['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena', 
                            'Bucaramanga', 'Pereira', 'Cúcuta', 'Otras']
                )
                
                form_data['estrato_socioeconomico'] = st.selectbox(
                    "Estrato socioeconómico:",
                    options=[1, 2, 3, 4, 5, 6]
                )
                
                form_data['personas_a_cargo'] = st.number_input(
                    "Personas a cargo:",
                    min_value=0,
                    max_value=10,
                    value=0
                )
        
        # ==================== TAB 2: DATOS LABORALES ====================
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                form_data['tipo_empleo'] = st.selectbox(
                    "Tipo de empleo:",
                    options=['Formal', 'Informal', 'Independiente']
                )
                
                form_data['antiguedad_empleo'] = st.number_input(
                    "Antigüedad en el empleo (años):",
                    min_value=0.0,
                    max_value=40.0,
                    value=5.0,
                    step=0.5
                )
            
            with col2:
                form_data['salario_mensual'] = st.number_input(
                    "Salario mensual (COP):",
                    min_value=1000000,
                    max_value=50000000,
                    value=3000000,
                    step=100000,
                    format="%d"
                )
                
                form_data['egresos_mensuales'] = st.number_input(
                    "Egresos mensuales (COP):",
                    min_value=500000,
                    max_value=30000000,
                    value=2000000,
                    step=100000,
                    format="%d"
                )
        
        # ==================== TAB 3: DATOS FINANCIEROS ====================
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                form_data['puntaje_datacredito'] = st.number_input(
                    "Puntaje DataCrédito:",
                    min_value=150,
                    max_value=950,
                    value=700,
                    help="Score crediticio entre 150 y 950"
                )
                
                form_data['patrimonio_total'] = st.number_input(
                    "Patrimonio total (COP):",
                    min_value=0,
                    max_value=5000000000,
                    value=50000000,
                    step=1000000,
                    format="%d"
                )
                
                form_data['numero_propiedades'] = st.number_input(
                    "Número de propiedades:",
                    min_value=0,
                    max_value=10,
                    value=0
                )
            
            with col2:
                form_data['saldo_promedio_banco'] = st.number_input(
                    "Saldo promedio banco (COP):",
                    min_value=0,
                    max_value=500000000,
                    value=5000000,
                    step=100000,
                    format="%d"
                )
                
                form_data['numero_demandas'] = st.number_input(
                    "Número de demandas legales:",
                    min_value=0,
                    max_value=10,
                    value=0
                )
        
        # ==================== TAB 4: DATOS DEL INMUEBLE ====================
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                form_data['valor_inmueble'] = st.number_input(
                    "Valor del inmueble (COP):",
                    min_value=20000000,
                    max_value=2000000000,
                    value=150000000,
                    step=5000000,
                    format="%d"
                )
                
                form_data['porcentaje_cuota_inicial'] = st.slider(
                    "Cuota inicial (%):",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="Porcentaje de cuota inicial"
                )
                
                form_data['anos_inmueble'] = st.number_input(
                    "Años del inmueble:",
                    min_value=0,
                    max_value=100,
                    value=5
                )
            
            with col2:
                form_data['plazo_credito'] = st.slider(
                    "Plazo del crédito (años):",
                    min_value=5,
                    max_value=30,
                    value=20
                )
                
                form_data['tasa_interes_anual'] = st.number_input(
                    "Tasa de interés anual (%):",
                    min_value=5.0,
                    max_value=25.0,
                    value=12.0,
                    step=0.1,
                    format="%.1f"
                )
        
        return form_data
    
    def calculate_derived_features(self, form_data: Dict) -> Dict:
        """
        Calcula características derivadas a partir de los datos del formulario
        
        Args:
            form_data: Datos del formulario
            
        Returns:
            Datos con características derivadas
        """
        data = form_data.copy()
        
        # Calcular características derivadas básicas
        data['valor_cuota_inicial'] = data['valor_inmueble'] * (data['porcentaje_cuota_inicial'] / 100)
        data['monto_credito'] = data['valor_inmueble'] - data['valor_cuota_inicial']
        
        # Calcular cuota mensual
        i = data['tasa_interes_anual'] / 12 / 100
        n = data['plazo_credito'] * 12
        
        if i > 0:
            data['cuota_mensual'] = data['monto_credito'] * (i * (1 + i)**n) / ((1 + i)**n - 1)
        else:
            data['cuota_mensual'] = data['monto_credito'] / n
        
        # Ratios importantes
        data['ltv'] = (data['monto_credito'] / data['valor_inmueble']) * 100
        data['dti'] = (data['cuota_mensual'] / data['salario_mensual']) * 100
        data['capacidad_ahorro'] = data['salario_mensual'] - data['egresos_mensuales']
        data['capacidad_residual'] = data['capacidad_ahorro'] - data['cuota_mensual']
        
        # Aplicar ingeniería de características completa
        try:
            # Crear DataFrame temporal
            temp_df = pd.DataFrame([data])
            
            # Aplicar ingeniería de características si está disponible
            if self.feature_engineer:
                engineer = self.feature_engineer(temp_df)
                enhanced_df = engineer.generate_all_features()
                
                # Convertir de vuelta a diccionario
                data = enhanced_df.iloc[0].to_dict()
            else:
                # Si no hay feature engineer, aplicar transformaciones básicas manualmente
                # Codificar variables categóricas
                categorical_mappings = {
                    'estado_civil': {'Soltero': 0, 'Casado': 1, 'Unión Libre': 2, 'Divorciado': 3, 'Viudo': 4},
                    'nivel_educacion': {'Bachiller': 0, 'Técnico': 1, 'Profesional': 2, 'Posgrado': 3},
                    'tipo_empleo': {'Informal': 0, 'Independiente': 1, 'Formal': 2},
                    'ciudad': {'Otras': 0, 'Cúcuta': 1, 'Pereira': 2, 'Bucaramanga': 3, 'Cartagena': 4,
                              'Barranquilla': 5, 'Cali': 6, 'Medellín': 7, 'Bogotá': 8}
                }
                
                for col, mapping in categorical_mappings.items():
                    if col in data:
                        data[f'{col}_encoded'] = mapping.get(data[col], 0)
                
                # Calcular características adicionales comunes
                data['ratio_patrimonio_ingreso'] = data.get('patrimonio_total', 0) / max(data['salario_mensual'], 1)
                data['ratio_saldo_ingreso'] = data.get('saldo_promedio_banco', 0) / max(data['salario_mensual'], 1)
                data['ratio_cuota_capacidad'] = data['cuota_mensual'] / max(data['capacidad_ahorro'], 1)
                
                # Ratios adicionales que el RBM espera
                data['ratio_cuota_ingreso'] = data['cuota_mensual'] / max(data['salario_mensual'], 1)
                data['ratio_cuota_ahorro'] = data['cuota_mensual'] / max(data['capacidad_ahorro'], 1)
                data['ratio_egreso_salario'] = data.get('egresos_mensuales', 0) / max(data['salario_mensual'], 1)
                data['ratio_patrimonio_deuda'] = data.get('patrimonio_total', 0) / max(data['monto_credito'], 1)
                
                # Meses de colchón
                data['meses_colchon'] = data.get('saldo_promedio_banco', 0) / max(data['cuota_mensual'], 1)
                
                # Scores de estabilidad (usar el mismo cálculo que FeatureEngineer)
                antiguedad = data.get('antiguedad_empleo', 0)
                tipo_empleo = data.get('tipo_empleo', '')
                
                # Score base por antigüedad
                score_estabilidad_laboral = min(100, antiguedad * 10)
                
                # Ajuste por tipo de empleo
                if tipo_empleo == "Formal":
                    score_estabilidad_laboral += 25
                elif tipo_empleo == "Independiente":
                    score_estabilidad_laboral += 10
                
                data['score_estabilidad_laboral'] = max(0, min(125, score_estabilidad_laboral))
                
                # Score de edad (mismo cálculo que FeatureEngineer)
                edad = data.get('edad', 0)
                if edad < 25:
                    data['score_edad'] = -30
                elif edad < 30:
                    data['score_edad'] = 10
                elif edad <= 55:
                    data['score_edad'] = 40
                else:
                    data['score_edad'] = max(-100, -8 * (edad - 55))
                
                # Flag sobreendeudamiento
                data['flag_sobreendeudamiento'] = 1 if data['dti'] > 40 else 0
                
                # Riesgo legal
                data['riesgo_legal'] = 100 * (1 - np.exp(-2 * data.get('numero_demandas', 0)))
                
                # Interacciones (NOMBRE CORRECTO que el RBM espera)
                data['edad_x_antiguedad'] = data.get('edad', 0) * data.get('antiguedad_empleo', 0)
                
                # Educación x Salario (necesita score_educacion primero)
                education_map = {'Bachiller': 1, 'Técnico': 2, 'Profesional': 3, 'Posgrado': 4}
                score_educacion = education_map.get(data.get('nivel_educacion', ''), 1)
                data['educacion_x_salario'] = score_educacion * (data['salario_mensual'] / 1000000)
                
                # LTV x Puntaje
                data['ltv_x_puntaje'] = data['ltv'] * (900 - data.get('puntaje_datacredito', 700)) / 100
                
                # Puntaje de riesgo (si no existe)
                if 'puntaje_riesgo' not in data:
                    data['puntaje_riesgo'] = 0
                
        except Exception as e:
            st.warning(f"⚠️ No se pudieron aplicar todas las características derivadas: {e}")
        
        return data
    
    def predict_risk(self, applicant_data: Dict) -> Dict:
        """
        Predice el riesgo crediticio
        
        Args:
            applicant_data: Datos del solicitante
            
        Returns:
            Resultados de la predicción
        """
        if not self.model_data:
            raise ValueError("No hay modelo cargado")
        
        # Preparar datos para predicción
        model = self.model_data['model']
        scaler = self.model_data['scaler']
        label_encoder = self.model_data['label_encoder']
        feature_names = self.model_data['feature_names']
        
        # Determinar si el modelo usa características RBM
        uses_rbm = any('RBM_H' in feat for feat in feature_names)
        
        # Crear DataFrame con datos del solicitante
        # IMPORTANTE: applicant_data ya tiene TODAS las características calculadas
        df_applicant = pd.DataFrame([applicant_data])
        
        # Debug: verificar qué características tenemos
        st.info(f"📊 Características disponibles en applicant_data: {len(applicant_data)} variables")
        
        # Si el modelo usa RBM, necesitamos transformar los datos ORIGINALES (no los calculados)
        if uses_rbm:
            st.info("🔄 Modelo entrenado con RBM detectado. Aplicando transformación...")
            # Cargar modelo RBM
            rbm_model_path = self._find_rbm_model()
            if rbm_model_path:
                try:
                    from src.rbm_model import RestrictedBoltzmannMachine
                    rbm = RestrictedBoltzmannMachine.load_model(rbm_model_path)
                    
                    # Obtener características originales que el RBM espera
                    rbm_feature_names = rbm.feature_names if hasattr(rbm, 'feature_names') else []
                    
                    if not rbm_feature_names:
                        st.warning("⚠️ El modelo RBM no tiene nombres de características guardados")
                        # Intentar usar características numéricas del applicant_data
                        rbm_feature_names = [k for k, v in applicant_data.items() if isinstance(v, (int, float))]
                    
                    # Preparar datos para RBM (solo características que el RBM conoce)
                    X_for_rbm = []
                    missing_features = []
                    for feat in rbm_feature_names:
                        if feat in applicant_data:
                            X_for_rbm.append(applicant_data[feat])
                        else:
                            X_for_rbm.append(0)
                            missing_features.append(feat)
                    
                    if missing_features and len(missing_features) <= 10:
                        st.info(f"ℹ️ Características para RBM faltantes (usando 0): {', '.join(missing_features[:10])}")
                    
                    X_for_rbm = np.array(X_for_rbm).reshape(1, -1)
                    
                    # Transformar con RBM
                    rbm_features = rbm.transform(X_for_rbm)
                    
                    # Crear nombres para características RBM
                    rbm_feature_names_new = [f"RBM_H{i+1}" for i in range(rbm_features.shape[1])]
                    df_rbm = pd.DataFrame(rbm_features, columns=rbm_feature_names_new)
                    
                    # IMPORTANTE: Cuando usamos RBM, el modelo supervisado fue entrenado con:
                    # características originales + características calculadas + características RBM
                    # Necesitamos combinar todo
                    df_applicant = pd.concat([df_applicant.reset_index(drop=True), df_rbm], axis=1)
                    
                    st.success(f"✅ Transformación RBM aplicada: {rbm_features.shape[1]} características latentes agregadas")
                    
                except Exception as e:
                    st.error(f"❌ Error aplicando transformación RBM: {e}")
                    st.exception(e)
                    # Continuar sin RBM, usando valores por defecto para características RBM
                    for feat in feature_names:
                        if 'RBM_H' in feat and feat not in df_applicant.columns:
                            df_applicant[feat] = 0
            else:
                st.warning("⚠️ No se encontró modelo RBM. Usando valores por defecto para características RBM.")
                # Crear características RBM con valores por defecto
                for feat in feature_names:
                    if 'RBM_H' in feat and feat not in df_applicant.columns:
                        df_applicant[feat] = 0
        
        # Crear DataFrame con las características requeridas por el modelo
        prediction_data = {}
        truly_missing_features = []
        found_in_applicant = []
        
        for feature in feature_names:
            if feature in df_applicant.columns:
                prediction_data[feature] = df_applicant[feature].iloc[0]
            elif feature in applicant_data:
                # Si no está en df_applicant pero sí en applicant_data original, usarlo
                prediction_data[feature] = applicant_data[feature]
                found_in_applicant.append(feature)
            else:
                # Valor por defecto para características faltantes
                prediction_data[feature] = 0
                truly_missing_features.append(feature)
        
        # Solo mostrar advertencia si REALMENTE faltan características
        if truly_missing_features:
            st.warning(f"⚠️ {len(truly_missing_features)} características faltantes (usando valor 0)")
            if len(truly_missing_features) <= 10:
                st.info(f"Características faltantes: {', '.join(truly_missing_features[:10])}")
        
        # Informar sobre características encontradas en applicant_data
        if found_in_applicant:
            st.success(f"✅ {len(found_in_applicant)} características recuperadas de applicant_data")
        
        # Convertir a DataFrame y escalar
        X_pred = pd.DataFrame([prediction_data])
        X_pred_scaled = scaler.transform(X_pred)
        
        # Realizar predicción
        prediction = model.predict(X_pred_scaled)[0]
        prediction_proba = model.predict_proba(X_pred_scaled)[0]
        
        # Decodificar predicción
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        # Crear diccionario de probabilidades (convertir a Python float inmediatamente)
        class_probabilities = {}
        for i, class_name in enumerate(label_encoder.classes_):
            prob_value = prediction_proba[i]
            # Convertir numpy float32 a Python float
            class_probabilities[class_name] = float(prob_value.item()) if hasattr(prob_value, 'item') else float(prob_value)
        
        # Generar explicación
        explanation = self._generate_explanation(applicant_data, predicted_class, class_probabilities)
        
        # Generar recomendación
        recommendation = self._generate_recommendation(predicted_class, class_probabilities, applicant_data)
        
        results = {
            'predicted_class': predicted_class,
            'probabilities': class_probabilities,
            'explanation': explanation,
            'recommendation': recommendation,
            'risk_factors': self._identify_risk_factors(applicant_data),
            'applicant_data': applicant_data
        }
        
        return results
    
    def _find_rbm_model(self) -> Optional[str]:
        """
        Busca el modelo RBM más reciente
        
        Returns:
            Ruta al modelo RBM o None
        """
        rbm_dir = "models/rbm"
        if not os.path.exists(rbm_dir):
            return None
        
        rbm_files = [f for f in os.listdir(rbm_dir) if f.endswith('.pkl')]
        if not rbm_files:
            return None
        
        # Ordenar por fecha de modificación (más reciente primero)
        rbm_files_with_time = [(f, os.path.getmtime(os.path.join(rbm_dir, f))) for f in rbm_files]
        rbm_files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        return os.path.join(rbm_dir, rbm_files_with_time[0][0])
    
    def _generate_explanation(self, data: Dict, prediction: str, probabilities: Dict) -> str:
        """Genera explicación en lenguaje natural"""
        
        # Factores principales
        factors = []
        
        # Puntaje DataCrédito
        puntaje = data.get('puntaje_datacredito', 0)
        if puntaje < 600:
            factors.append(f"puntaje DataCrédito bajo ({puntaje})")
        elif puntaje > 750:
            factors.append(f"excelente puntaje DataCrédito ({puntaje})")
        
        # DTI
        dti = data.get('dti', 0)
        if dti > 35:
            factors.append(f"alto ratio de endeudamiento ({dti:.1f}%)")
        elif dti < 25:
            factors.append(f"bajo ratio de endeudamiento ({dti:.1f}%)")
        
        # Capacidad residual
        cap_residual = data.get('capacidad_residual', 0)
        if cap_residual < 0:
            factors.append("capacidad residual negativa")
        elif cap_residual > 500000:
            factors.append("buena capacidad residual")
        
        # Estabilidad laboral
        if data.get('tipo_empleo') == 'Formal' and data.get('antiguedad_empleo', 0) > 3:
            factors.append("empleo formal estable")
        elif data.get('tipo_empleo') == 'Informal':
            factors.append("empleo informal")
        
        # Construir explicación
        confidence = max(probabilities.values())
        
        explanation = f"El solicitante presenta riesgo **{prediction.upper()}** con {confidence:.1%} de confianza. "
        
        if factors:
            explanation += f"Esto se debe principalmente a: {', '.join(factors)}."
        
        return explanation
    
    def _generate_recommendation(self, prediction: str, probabilities: Dict, data: Dict) -> Dict:
        """Genera recomendación de aprobación"""
        
        confidence = max(probabilities.values())
        
        if prediction == 'Bajo' and confidence > 0.7:
            decision = "APROBAR"
            color = "success"
            icon = "✅"
        elif prediction == 'Alto' or confidence > 0.8:
            decision = "RECHAZAR"
            color = "error"
            icon = "❌"
        else:
            decision = "REVISAR MANUALMENTE"
            color = "warning"
            icon = "⚠️"
        
        # Condiciones adicionales
        conditions = []
        
        if data.get('dti', 0) > 40:
            conditions.append("DTI superior al 40%")
        
        if data.get('numero_demandas', 0) > 0:
            conditions.append("Historial de demandas legales")
        
        if data.get('capacidad_residual', 0) < 0:
            conditions.append("Capacidad residual negativa")
        
        return {
            'decision': decision,
            'color': color,
            'icon': icon,
            'confidence': confidence,
            'conditions': conditions
        }
    
    def _identify_risk_factors(self, data: Dict) -> List[Dict]:
        """Identifica los principales factores de riesgo"""
        
        risk_factors = []
        
        # Factor 1: Puntaje DataCrédito
        puntaje = data.get('puntaje_datacredito', 0)
        if puntaje < 600:
            impact = "ALTO"
            direction = "Aumenta"
        elif puntaje > 750:
            impact = "BAJO"
            direction = "Disminuye"
        else:
            impact = "MEDIO"
            direction = "Neutral"
        
        risk_factors.append({
            'factor': 'Puntaje DataCrédito',
            'value': puntaje,
            'impact': impact,
            'direction': direction
        })
        
        # Factor 2: DTI
        dti = data.get('dti', 0)
        if dti > 35:
            impact = "ALTO"
            direction = "Aumenta"
        elif dti < 25:
            impact = "BAJO"
            direction = "Disminuye"
        else:
            impact = "MEDIO"
            direction = "Neutral"
        
        risk_factors.append({
            'factor': 'Ratio Deuda/Ingreso (DTI)',
            'value': f"{dti:.1f}%",
            'impact': impact,
            'direction': direction
        })
        
        # Factor 3: Capacidad residual
        cap_residual = data.get('capacidad_residual', 0)
        if cap_residual < 0:
            impact = "ALTO"
            direction = "Aumenta"
        elif cap_residual > 500000:
            impact = "BAJO"
            direction = "Disminuye"
        else:
            impact = "MEDIO"
            direction = "Neutral"
        
        risk_factors.append({
            'factor': 'Capacidad Residual',
            'value': f"${cap_residual:,.0f}",
            'impact': impact,
            'direction': direction
        })
        
        # Factor 4: LTV
        ltv = data.get('ltv', 0)
        if ltv > 85:
            impact = "ALTO"
            direction = "Aumenta"
        elif ltv < 70:
            impact = "BAJO"
            direction = "Disminuye"
        else:
            impact = "MEDIO"
            direction = "Neutral"
        
        risk_factors.append({
            'factor': 'Loan-to-Value (LTV)',
            'value': f"{ltv:.1f}%",
            'impact': impact,
            'direction': direction
        })
        
        # Factor 5: Estabilidad laboral
        tipo_empleo = data.get('tipo_empleo', '')
        antiguedad = data.get('antiguedad_empleo', 0)
        
        if tipo_empleo == 'Formal' and antiguedad > 3:
            impact = "BAJO"
            direction = "Disminuye"
        elif tipo_empleo == 'Informal' or antiguedad < 1:
            impact = "ALTO"
            direction = "Aumenta"
        else:
            impact = "MEDIO"
            direction = "Neutral"
        
        risk_factors.append({
            'factor': 'Estabilidad Laboral',
            'value': f"{tipo_empleo}, {antiguedad} años",
            'impact': impact,
            'direction': direction
        })
        
        return risk_factors

def render_prediction_interface():
    """Renderiza la interfaz de predicción en Streamlit"""
    st.title("🔮 Predicción de Riesgo Crediticio")
    st.markdown("### *Evalúa el riesgo de nuevos solicitantes*")
    
    # Crear predictor
    predictor = CreditRiskPredictor()
    
    # Verificar modelos disponibles
    if not predictor.available_models:
        st.error("❌ No hay modelos entrenados disponibles. Ve a 'Modelos Supervisados' primero.")
        return
    
    # Selección de modelo
    st.subheader("🤖 Selección de Modelo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model_key = st.selectbox(
            "Selecciona el modelo para predicción:",
            options=list(predictor.available_models.keys()),
            format_func=lambda x: predictor.available_models[x]['name']
        )
    
    with col2:
        if selected_model_key:
            metrics = predictor.available_models[selected_model_key]['metrics']
            if metrics:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                st.metric("F1-Score", f"{metrics.get('f1_weighted', 0):.3f}")
    
    # Cargar modelo seleccionado
    if selected_model_key and predictor.load_model(selected_model_key):
        st.success(f"✅ Modelo cargado: {predictor.available_models[selected_model_key]['name']}")
        
        # Formulario de predicción
        form_data = predictor.create_prediction_form()
        
        # Validaciones en tiempo real
        st.subheader("✅ Validaciones")
        
        validations = []
        
        # Calcular monto_credito para validaciones
        valor_cuota_inicial = form_data['valor_inmueble'] * (form_data['porcentaje_cuota_inicial'] / 100)
        monto_credito = form_data['valor_inmueble'] - valor_cuota_inicial
        
        # Validación 1: DTI calculado
        if form_data['salario_mensual'] > 0:
            cuota_estimada = form_data['valor_inmueble'] * 0.8 * 0.01  # Estimación rápida
            dti_estimado = (cuota_estimada / form_data['salario_mensual']) * 100
            
            if dti_estimado > 40:
                validations.append(("⚠️", f"DTI estimado alto: {dti_estimado:.1f}%"))
            else:
                validations.append(("✅", f"DTI estimado aceptable: {dti_estimado:.1f}%"))
        
        # Validación 2: Capacidad de ahorro
        capacidad_ahorro = form_data['salario_mensual'] - form_data['egresos_mensuales']
        if capacidad_ahorro <= 0:
            validations.append(("❌", "Capacidad de ahorro negativa"))
        else:
            validations.append(("✅", f"Capacidad de ahorro: ${capacidad_ahorro:,.0f}"))
        
        # Validación 3: Consistencia monto vs valor
        if monto_credito > form_data['valor_inmueble']:
            validations.append(("❌", "Monto crédito > Valor inmueble"))
        else:
            validations.append(("✅", "Monto crédito consistente"))
        
        # Mostrar validaciones
        for icon, message in validations:
            if icon == "❌":
                st.error(f"{icon} {message}")
            elif icon == "⚠️":
                st.warning(f"{icon} {message}")
            else:
                st.success(f"{icon} {message}")
        
        # Botón de predicción
        if st.button("🎯 PREDECIR RIESGO", type="primary", use_container_width=True):
            with st.spinner("🔮 Analizando riesgo crediticio..."):
                try:
                    # Calcular características derivadas
                    enhanced_data = predictor.calculate_derived_features(form_data)
                    
                    # Realizar predicción
                    prediction_results = predictor.predict_risk(enhanced_data)
                    
                    # Mostrar resultados
                    st.divider()
                    st.subheader("🎯 Resultados de la Predicción")
                    
                    # Predicción principal
                    predicted_class = prediction_results['predicted_class']
                    probabilities = prediction_results['probabilities']
                    
                    # Color según riesgo
                    if predicted_class == 'Bajo':
                        risk_color = '#28a745'
                        risk_emoji = '🟢'
                    elif predicted_class == 'Medio':
                        risk_color = '#ffc107'
                        risk_emoji = '🟡'
                    else:
                        risk_color = '#dc3545'
                        risk_emoji = '🔴'
                    
                    # Mostrar predicción principal
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {risk_color}20; border: 2px solid {risk_color};">
                        <h2 style="color: {risk_color}; margin: 0;">{risk_emoji} RIESGO {predicted_class.upper()}</h2>
                        <p style="font-size: 18px; margin: 10px 0;">Confianza: {max(probabilities.values()):.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probabilidades por clase
                    st.subheader("📊 Probabilidades por Clase")
                    
                    for class_name, prob in probabilities.items():
                        # Barra de progreso visual
                        if class_name == 'Bajo':
                            color = '#28a745'
                            emoji = '🟢'
                        elif class_name == 'Medio':
                            color = '#ffc107'
                            emoji = '🟡'
                        else:
                            color = '#dc3545'
                            emoji = '🔴'
                        
                        st.markdown(f"""
                        **{emoji} {class_name}:** {prob:.1%}
                        """)
                        st.progress(prob)
                    
                    # Gráfico de probabilidades
                    fig_probs = px.bar(
                        x=list(probabilities.keys()),
                        y=list(probabilities.values()),
                        title="Distribución de Probabilidades",
                        color=list(probabilities.values()),
                        color_continuous_scale=['#28a745', '#ffc107', '#dc3545']
                    )
                    
                    fig_probs.update_layout(
                        template="plotly_white",
                        height=400,
                        showlegend=False,
                        yaxis_title="Probabilidad"
                    )
                    
                    st.plotly_chart(fig_probs, use_container_width=True)
                    
                    # Factores de riesgo
                    st.subheader("⚠️ Análisis de Factores de Riesgo")
                    
                    risk_factors = prediction_results['risk_factors']
                    
                    for factor in risk_factors:
                        if factor['impact'] == 'ALTO':
                            st.error(f"🔴 **{factor['factor']}:** {factor['value']} - {factor['direction']} riesgo")
                        elif factor['impact'] == 'MEDIO':
                            st.warning(f"🟡 **{factor['factor']}:** {factor['value']} - {factor['direction']} riesgo")
                        else:
                            st.success(f"🟢 **{factor['factor']}:** {factor['value']} - {factor['direction']} riesgo")
                    
                    # Recomendación final
                    st.subheader("💼 Recomendación")
                    
                    recommendation = prediction_results['recommendation']
                    
                    if recommendation['color'] == 'success':
                        st.success(f"{recommendation['icon']} **{recommendation['decision']}**")
                    elif recommendation['color'] == 'error':
                        st.error(f"{recommendation['icon']} **{recommendation['decision']}**")
                    else:
                        st.warning(f"{recommendation['icon']} **{recommendation['decision']}**")
                    
                    # Condiciones adicionales
                    if recommendation['conditions']:
                        st.markdown("**Condiciones a considerar:**")
                        for condition in recommendation['conditions']:
                            st.markdown(f"- {condition}")
                    
                    # Explicación detallada
                    st.subheader("📝 Explicación Detallada")
                    st.markdown(prediction_results['explanation'])
                    
                    # Guardar predicción en historial
                    _save_prediction_to_history(predictor, prediction_results)
                    
                except Exception as e:
                    st.error(f"❌ Error realizando predicción: {e}")
                    st.exception(e)
    
def _save_prediction_to_history(predictor: CreditRiskPredictor, prediction_results: Dict):
    """Guarda la predicción en el historial"""
    try:
        history_path = "data/predictions_history.json"
        
        # Cargar historial existente
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Agregar nueva predicción
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'model_used': predictor.selected_model,
            'prediction': prediction_results['predicted_class'],
            'probabilities': prediction_results['probabilities'],
            'recommendation': prediction_results['recommendation']['decision'],
            'applicant_summary': {
                'edad': prediction_results['applicant_data'].get('edad'),
                'salario': prediction_results['applicant_data'].get('salario_mensual'),
                'puntaje_datacredito': prediction_results['applicant_data'].get('puntaje_datacredito'),
                'dti': prediction_results['applicant_data'].get('dti')
            }
        }
        
        history.append(prediction_record)
        
        # Mantener solo últimas 100 predicciones
        if len(history) > 100:
            history = history[-100:]
        
        # Guardar historial actualizado
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        st.warning(f"⚠️ No se pudo guardar en historial: {e}")

def render_prediction_module():
    """Función principal para renderizar el módulo de predicción"""
    render_prediction_interface()

if __name__ == "__main__":
    print("Módulo de predicción cargado correctamente")