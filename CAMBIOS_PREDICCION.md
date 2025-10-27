# Correcciones al Módulo de Predicción

## Problema Identificado

Las predicciones no cambiaban cuando se variaban los valores de entrada porque:

1. **Falta de transformación de características**: El módulo no aplicaba las mismas transformaciones que se usaron durante el entrenamiento
2. **No se distinguía entre modelos**: No había forma de saber si un modelo fue entrenado con datos normales o con características RBM
3. **Características faltantes**: Cuando faltaban características, se usaba valor 0 sin aplicar las transformaciones necesarias

## Soluciones Implementadas

### 1. Mejora en `calculate_derived_features()` (líneas 290-348)

**Antes:**
- Solo calculaba características básicas (LTV, DTI, capacidad residual)
- Dependía completamente del FeatureEngineer que podía no estar disponible

**Ahora:**
- Calcula todas las características derivadas básicas
- Si no hay FeatureEngineer, aplica transformaciones manualmente:
  - Codificación de variables categóricas (estado_civil, nivel_educacion, tipo_empleo, ciudad)
  - Ratios adicionales (patrimonio/ingreso, saldo/ingreso, cuota/capacidad)
  - Score de estabilidad laboral
- Garantiza que las características estén en el mismo formato que durante el entrenamiento

### 2. Detección automática de modelos RBM (líneas 359-420)

**Nueva funcionalidad:**
- Detecta automáticamente si el modelo usa características RBM (busca 'RBM_H' en feature_names)
- Si usa RBM:
  1. Carga el modelo RBM más reciente
  2. Extrae las características que el RBM espera
  3. Transforma los datos de entrada usando el RBM
  4. Combina características originales + características latentes RBM
- Si no encuentra el modelo RBM, usa valores por defecto y muestra advertencia

### 3. Nuevo método `_find_rbm_model()` (líneas 462-481)

**Funcionalidad:**
- Busca modelos RBM en el directorio `models/rbm/`
- Retorna el modelo más reciente (por fecha de modificación)
- Maneja casos donde no hay modelos RBM disponibles

### 4. Mejor manejo de características faltantes (líneas 422-435)

**Mejoras:**
- Registra todas las características faltantes
- Muestra advertencias informativas al usuario
- Limita la lista mostrada a 10 características para no saturar la UI
- Usa valor 0 como fallback pero informa al usuario

## Flujo de Predicción Corregido

### Para Modelos con Datos Normales:

```
Datos del formulario
    ↓
calculate_derived_features()
    ↓ (calcula LTV, DTI, ratios, codifica categóricas)
Datos enriquecidos
    ↓
Alinear con feature_names del modelo
    ↓
Escalar con el scaler del modelo
    ↓
Predicción
```

### Para Modelos con RBM:

```
Datos del formulario
    ↓
calculate_derived_features()
    ↓ (calcula características básicas)
Datos enriquecidos
    ↓
Cargar modelo RBM
    ↓
Extraer características que RBM espera
    ↓
Transformar con RBM → características latentes
    ↓
Combinar datos originales + características RBM
    ↓
Alinear con feature_names del modelo
    ↓
Escalar con el scaler del modelo
    ↓
Predicción
```

## Cómo Probar las Correcciones

### Prueba 1: Modelo con Datos Normales

1. Ir al módulo de "Predicción"
2. Seleccionar un modelo entrenado con datos normales (ej: logistic_model, random_forest_model)
3. Ingresar datos de prueba:
   - Edad: 30, Salario: 3,000,000, DataCrédito: 700
4. Hacer predicción y anotar el resultado
5. Cambiar valores significativamente:
   - Edad: 30, Salario: 3,000,000, DataCrédito: 400
6. Hacer nueva predicción
7. **Verificar**: El riesgo debe cambiar (de Bajo a Alto probablemente)

### Prueba 2: Modelo con RBM

1. Primero entrenar un modelo con características RBM:
   - Ir a "RBM" y entrenar un modelo
   - Extraer características RBM
   - Ir a "Modelos Supervisados" y entrenar con el dataset "Con Características RBM"
2. Ir a "Predicción" y seleccionar el modelo entrenado con RBM
3. Ingresar datos de prueba y verificar que:
   - Se muestra mensaje "✅ Transformación RBM aplicada: X características latentes"
   - La predicción cambia cuando se modifican los valores de entrada

### Prueba 3: Sensibilidad a Cambios

Probar cambios en diferentes variables y verificar que el riesgo cambia apropiadamente:

| Variable | Valor Bajo Riesgo | Valor Alto Riesgo | Efecto Esperado |
|----------|-------------------|-------------------|-----------------|
| puntaje_datacredito | 800 | 400 | Riesgo aumenta significativamente |
| salario_mensual | 8,000,000 | 1,500,000 | Riesgo aumenta |
| dti (calculado) | 20% | 50% | Riesgo aumenta |
| ltv (calculado) | 60% | 95% | Riesgo aumenta |
| antiguedad_empleo | 10 años | 0.5 años | Riesgo aumenta |
| tipo_empleo | Formal | Informal | Riesgo aumenta |

## Mensajes de Diagnóstico

El módulo ahora muestra mensajes informativos:

- ✅ **Éxito**: "Transformación RBM aplicada: X características latentes"
- ⚠️ **Advertencia**: "X características faltantes (usando valor 0)"
- ℹ️ **Info**: "Características RBM faltantes (usando 0): X"
- ❌ **Error**: "Error aplicando transformación RBM: [detalle]"

Estos mensajes ayudan a diagnosticar problemas durante la predicción.

## Archivos Modificados

- `src/prediction.py`: Módulo principal de predicción (líneas 290-481)

## Notas Importantes

1. **Compatibilidad**: Los cambios son retrocompatibles con modelos existentes
2. **Fallback**: Si algo falla, el sistema usa valores por defecto y continúa
3. **Transparencia**: El usuario siempre ve qué transformaciones se aplicaron
4. **Robustez**: Maneja casos donde faltan características o modelos RBM

## Próximos Pasos Recomendados

1. Probar con diferentes combinaciones de valores de entrada
2. Verificar que las probabilidades sumen 1.0
3. Comparar predicciones con datos de entrenamiento conocidos
4. Documentar rangos esperados de probabilidades para cada clase de riesgo