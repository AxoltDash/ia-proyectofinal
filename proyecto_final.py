"""
PROYECTO FINAL: CLASIFICACIÓN DE HONGOS (VENENOSOS vs COMESTIBLES)
Facultad de Ciencias, UNAM - Aprendizaje de Máquina
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from ucimlrepo import fetch_ucirepo
import os
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Crear directorio de outputs si no existe
if not os.path.exists('outputs'):
    os.makedirs('outputs')

print("="*80)
print("PROYECTO FINAL: CLASIFICACIÓN DE HONGOS (VENENOSOS vs COMESTIBLES)")
print("="*80)

# ============================================================================
# 1. DESCARGA DEL DATASET
# ============================================================================
print("\n[1] DESCARGANDO DATASET...")
print("-"*80)

try:
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features
    y = mushroom.data.targets
    print(f"✓ Dataset descargado exitosamente desde UCI Machine Learning Repository")
except Exception as e:
    print(f"✗ Error al descargar dataset: {e}")
    exit()

# Confirmar que es un problema de clasificación binaria
print(f"\nProblema identificado: CLASIFICACIÓN BINARIA")
print(f"Clases a predecir: {y.columns[0]}")
print(f"Valores únicos en target: {y[y.columns[0]].unique()}")
print(f"Número de clases: {y[y.columns[0]].nunique()}")

# ============================================================================
# 2. DESCRIPCIÓN DEL DATASET
# ============================================================================
print("\n[2] DESCRIPCIÓN DEL DATASET")
print("-"*80)

# Obtener nombre de la columna target
target_col = y.columns[0]
y_series = y[target_col]

# Número de muestras y features
n_samples, n_features = X.shape
print(f"\nNúmero total de muestras: {n_samples}")
print(f"Número de features: {n_features}")
print(f"Features: {list(X.columns)}")

# Split 70% train, 15% validación, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_series, test_size=0.15, random_state=42, stratify=y_series
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp
)

print(f"\nDivisión del dataset:")
print(f"  • Entrenamiento: {len(X_train)} muestras (70%)")
print(f"  • Validación: {len(X_val)} muestras (15%)")
print(f"  • Test: {len(X_test)} muestras (15%)")

# Estadísticas descriptivas por columna (value_counts para categóricas)
print(f"\nEstadísticas descriptivas del dataset completo:")
print(f"\nDistribución de la variable objetivo ('{target_col}'):")
print(y_series.value_counts())
print(f"\nProporción:")
print(y_series.value_counts(normalize=True))

print("\nFrecuencia de las primeras 3 features:")
for col in X.columns[:3]:
    print(f"\n{col}:")
    print(X[col].value_counts().head())

# ============================================================================
# VISUALIZACIONES
# ============================================================================

# Visualización 1: Distribución de clases
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica 1: Distribución absoluta
y_counts = y_series.value_counts()
colors = ['#FF6B6B', '#4ECDC4']
axes[0].bar(y_counts.index, y_counts.values, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_title('Distribución de Clases (Valores Absolutos)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Clase', fontsize=10)
axes[0].set_ylabel('Cantidad de muestras', fontsize=10)
for i, v in enumerate(y_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Gráfica 2: Distribución proporcional
y_prop = y_series.value_counts(normalize=True) * 100
axes[1].pie(y_prop.values, labels=y_prop.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1].set_title('Distribución de Clases (Porcentaje)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/01_distribucion_clases.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfica guardada: outputs/01_distribucion_clases.png")
plt.close()

# Visualización 2: Frecuencia de features relevantes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

feature_samples = X.columns[:4]
for idx, col in enumerate(feature_samples):
    counts = X[col].value_counts()
    axes[idx].barh(range(len(counts)), counts.values, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_yticks(range(len(counts)))
    axes[idx].set_yticklabels(counts.index)
    axes[idx].set_title(f'Frecuencia: {col}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Cantidad', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/02_frecuencia_features.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada: outputs/02_frecuencia_features.png")
plt.close()

# ============================================================================
# 3. PREPARACIÓN DE DATOS: CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================================
print("\n[3] PREPARACIÓN DE DATOS")
print("-"*80)

# Codificar variables categóricas
label_encoders = {}

# Codificar X_train
X_train_encoded = X_train.copy()
for col in X_train_encoded.columns:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train_encoded[col])
    label_encoders[col] = le

# Codificar X_val y X_test usando los encoders del training
X_val_encoded = X_val.copy()
X_test_encoded = X_test.copy()
for col in X_val_encoded.columns:
    X_val_encoded[col] = label_encoders[col].transform(X_val_encoded[col])
    X_test_encoded[col] = label_encoders[col].transform(X_test_encoded[col])

# Codificar variable objetivo
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)
y_val_encoded = le_target.transform(y_val)
y_test_encoded = le_target.transform(y_test)

print(f"✓ Variables categóricas codificadas usando LabelEncoder")
print(f"✓ Clases codificadas: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# ============================================================================
# 4. ELECCIÓN DE MODELOS Y JUSTIFICACIÓN
# ============================================================================
print("\n[3.1] JUSTIFICACIÓN DE MODELOS")
print("-"*80)

justificacion = """
Se seleccionan dos modelos basados en árboles de decisión:

1. ÁRBOL DE DECISIÓN (DecisionTreeClassifier):
   - Interpretable y fácil de visualizar
   - Apropiado para datos categóricos
   - Rápido en entrenamiento
   - Útil para identificar features importantes
   - Riesgo de overfitting que mitiga con hiperparámetros

2. RANDOM FOREST (RandomForestClassifier):
   - Ensemble que reduce overfitting del árbol único
   - Mejor generalización en datos complejos
   - Maneja relaciones no-lineales
   - Proporciona importancia de features más robusta
   - Requiere más recursos pero mejor precisión esperada

Ambos modelos son ideales para problemas de clasificación binaria con features
categóricas, como el de hongos venenosos vs comestibles.
"""
print(justificacion)

# ============================================================================
# 5. ENTRENAMIENTO DETALLADO CON VALIDACIÓN
# ============================================================================
print("\n[4] ENTRENAMIENTO DE MODELOS CON AJUSTE DE HIPERPARÁMETROS")
print("-"*80)

# Diccionario para almacenar resultados
resultados = {}

# -------- MODELO 1: ÁRBOL DE DECISIÓN --------
print("\n[4.1] ÁRBOL DE DECISIÓN")
print("·"*40)

# Búsqueda de hiperparámetros óptimos en validación
print("Ajustando hiperparámetros en conjunto de validación...")

best_score_dt = 0
best_params_dt = {}
param_grid_dt = {
    'max_depth': [5, 7, 10, 15, 20],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10]
}

for max_depth in param_grid_dt['max_depth']:
    for criterion in param_grid_dt['criterion']:
        for min_samples_split in param_grid_dt['min_samples_split']:
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=criterion,
                min_samples_split=min_samples_split,
                random_state=42
            )
            dt.fit(X_train_encoded, y_train_encoded)
            score = dt.score(X_val_encoded, y_val_encoded)

            if score > best_score_dt:
                best_score_dt = score
                best_params_dt = {
                    'max_depth': max_depth,
                    'criterion': criterion,
                    'min_samples_split': min_samples_split
                }

# Entrenar modelo final con mejores parámetros
dt_model = DecisionTreeClassifier(
    max_depth=best_params_dt['max_depth'],
    criterion=best_params_dt['criterion'],
    min_samples_split=best_params_dt['min_samples_split'],
    random_state=42
)
dt_model.fit(X_train_encoded, y_train_encoded)

print(f"\nHiperparámetros óptimos encontrados:")
print(f"  • max_depth: {best_params_dt['max_depth']}")
print(f"  • criterion: {best_params_dt['criterion']}")
print(f"  • min_samples_split: {best_params_dt['min_samples_split']}")
print(f"  • Exactitud en validación: {best_score_dt:.4f}")

# Predicciones
y_train_pred_dt = dt_model.predict(X_train_encoded)
y_val_pred_dt = dt_model.predict(X_val_encoded)
y_test_pred_dt = dt_model.predict(X_test_encoded)

# -------- MODELO 2: RANDOM FOREST --------
print("\n[4.2] RANDOM FOREST")
print("·"*40)

print("Ajustando hiperparámetros en conjunto de validación...")

best_score_rf = 0
best_params_rf = {}
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'criterion': ['gini', 'entropy']
}

for n_estimators in param_grid_rf['n_estimators']:
    for max_depth in param_grid_rf['max_depth']:
        for criterion in param_grid_rf['criterion']:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                criterion=criterion,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_encoded, y_train_encoded)
            score = rf.score(X_val_encoded, y_val_encoded)

            if score > best_score_rf:
                best_score_rf = score
                best_params_rf = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'criterion': criterion
                }

# Entrenar modelo final con mejores parámetros
rf_model = RandomForestClassifier(
    n_estimators=best_params_rf['n_estimators'],
    max_depth=best_params_rf['max_depth'],
    criterion=best_params_rf['criterion'],
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_encoded, y_train_encoded)

print(f"\nHiperparámetros óptimos encontrados:")
print(f"  • n_estimators: {best_params_rf['n_estimators']}")
print(f"  • max_depth: {best_params_rf['max_depth']}")
print(f"  • criterion: {best_params_rf['criterion']}")
print(f"  • Exactitud en validación: {best_score_rf:.4f}")

# Predicciones
y_train_pred_rf = rf_model.predict(X_train_encoded)
y_val_pred_rf = rf_model.predict(X_val_encoded)
y_test_pred_rf = rf_model.predict(X_test_encoded)

# ============================================================================
# 6. EVALUACIÓN CON MÉTRICAS
# ============================================================================
print("\n[5] EVALUACIÓN Y COMPARACIÓN DE MODELOS")
print("-"*80)

def obtener_metricas(y_true, y_pred, modelo_nombre, conjunto_nombre):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'modelo': modelo_nombre,
        'conjunto': conjunto_nombre,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

# Calcular métricas para ambos modelos en test
metricas_dt = obtener_metricas(y_test_encoded, y_test_pred_dt, 'Árbol de Decisión', 'Test')
metricas_rf = obtener_metricas(y_test_encoded, y_test_pred_rf, 'Random Forest', 'Test')

print("\nMÉTRICAS EN CONJUNTO DE TEST:")
print("="*60)
print(f"\n{'ÁRBOL DE DECISIÓN':^60}")
print("-"*60)
print(f"Exactitud (Accuracy):  {metricas_dt['accuracy']:.4f} ({metricas_dt['accuracy']*100:.2f}%)")
print(f"Precisión:             {metricas_dt['precision']:.4f}")
print(f"Recall:                {metricas_dt['recall']:.4f}")
print(f"F1-Score:              {metricas_dt['f1']:.4f}")

print(f"\n{'RANDOM FOREST':^60}")
print("-"*60)
print(f"Exactitud (Accuracy):  {metricas_rf['accuracy']:.4f} ({metricas_rf['accuracy']*100:.2f}%)")
print(f"Precisión:             {metricas_rf['precision']:.4f}")
print(f"Recall:                {metricas_rf['recall']:.4f}")
print(f"F1-Score:              {metricas_rf['f1']:.4f}")

# Matrices de confusión
cm_dt = confusion_matrix(y_test_encoded, y_test_pred_dt)
cm_rf = confusion_matrix(y_test_encoded, y_test_pred_rf)

print("\nMATRICES DE CONFUSIÓN:")
print("="*60)
print(f"\n{'ÁRBOL DE DECISIÓN':^60}")
print("-"*60)
print(cm_dt)
print(f"\n{'RANDOM FOREST':^60}")
print("-"*60)
print(cm_rf)

# Visualización: Comparación de métricas lado a lado
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Métricas en barras
metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
valores_dt = [metricas_dt['accuracy'], metricas_dt['precision'],
              metricas_dt['recall'], metricas_dt['f1']]
valores_rf = [metricas_rf['accuracy'], metricas_rf['precision'],
              metricas_rf['recall'], metricas_rf['f1']]

x = np.arange(len(metricas_nombres))
width = 0.35

axes[0].bar(x - width/2, valores_dt, width, label='Árbol de Decisión',
            color='#FF6B6B', alpha=0.8, edgecolor='black')
axes[0].bar(x + width/2, valores_rf, width, label='Random Forest',
            color='#4ECDC4', alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Puntuación', fontsize=11, fontweight='bold')
axes[0].set_title('Comparación de Métricas en Test', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metricas_nombres, fontsize=10)
axes[0].legend(fontsize=10)
axes[0].set_ylim([0, 1.05])
axes[0].grid(axis='y', alpha=0.3)

# Matrices de confusión
axes[1].axis('off')
axes[1].text(0.5, 0.95, 'Matrices de Confusión (Test Set)',
             ha='center', fontsize=12, fontweight='bold', transform=axes[1].transAxes)

# Texto de matrices
texto_cm = f"""
ÁRBOL DE DECISIÓN:
{cm_dt[0][0]:5d}  {cm_dt[0][1]:5d}
{cm_dt[1][0]:5d}  {cm_dt[1][1]:5d}

RANDOM FOREST:
{cm_rf[0][0]:5d}  {cm_rf[0][1]:5d}
{cm_rf[1][0]:5d}  {cm_rf[1][1]:5d}
"""
axes[1].text(0.1, 0.5, texto_cm, fontfamily='monospace', fontsize=10,
             verticalalignment='center', transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig('outputs/03_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfica guardada: outputs/03_comparacion_modelos.png")
plt.close()

# Visualización: Matrices de confusión heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Comestible', 'Venenoso'],
            yticklabels=['Comestible', 'Venenoso'],
            cbar_kws={'label': 'Cantidad'})
axes[0].set_title('Matriz de Confusión - Árbol de Decisión', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Real', fontsize=10)
axes[0].set_xlabel('Predicción', fontsize=10)

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Comestible', 'Venenoso'],
            yticklabels=['Comestible', 'Venenoso'],
            cbar_kws={'label': 'Cantidad'})
axes[1].set_title('Matriz de Confusión - Random Forest', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Real', fontsize=10)
axes[1].set_xlabel('Predicción', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/04_matrices_confusion.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada: outputs/04_matrices_confusion.png")
plt.close()

# ============================================================================
# 7. IMPORTANCIA DE FEATURES
# ============================================================================
print("\n[5.1] IMPORTANCIA DE FEATURES")
print("-"*80)

# Features importances
importances_dt = dt_model.feature_importances_
importances_rf = rf_model.feature_importances_

# Ordenar por importancia
indices_dt = np.argsort(importances_dt)[::-1][:10]
indices_rf = np.argsort(importances_rf)[::-1][:10]

print("\nTOP 10 FEATURES - ÁRBOL DE DECISIÓN:")
for i, idx in enumerate(indices_dt, 1):
    print(f"  {i:2d}. {X.columns[idx]:25s} : {importances_dt[idx]:.4f}")

print("\nTOP 10 FEATURES - RANDOM FOREST:")
for i, idx in enumerate(indices_rf, 1):
    print(f"  {i:2d}. {X.columns[idx]:25s} : {importances_rf[idx]:.4f}")

# Visualización: Feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Árbol de Decisión
top_features_dt = [X.columns[i] for i in indices_dt]
axes[0].barh(range(len(top_features_dt)), importances_dt[indices_dt],
             color='#FF6B6B', alpha=0.7, edgecolor='black')
axes[0].set_yticks(range(len(top_features_dt)))
axes[0].set_yticklabels(top_features_dt)
axes[0].set_xlabel('Importancia', fontsize=10, fontweight='bold')
axes[0].set_title('Top 10 Features - Árbol de Decisión', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()

# Random Forest
top_features_rf = [X.columns[i] for i in indices_rf]
axes[1].barh(range(len(top_features_rf)), importances_rf[indices_rf],
             color='#4ECDC4', alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(top_features_rf)))
axes[1].set_yticklabels(top_features_rf)
axes[1].set_xlabel('Importancia', fontsize=10, fontweight='bold')
axes[1].set_title('Top 10 Features - Random Forest', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/05_importancia_features.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfica guardada: outputs/05_importancia_features.png")
plt.close()

# ============================================================================
# 8. CONCLUSIONES Y ANÁLISIS CUALITATIVO
# ============================================================================
print("\n[6] CONCLUSIONES Y ANÁLISIS")
print("="*80)

mejor_modelo = "Random Forest" if metricas_rf['f1'] > metricas_dt['f1'] else "Árbol de Decisión"
diferencia_f1 = abs(metricas_rf['f1'] - metricas_dt['f1'])

conclusiones = f"""
ANÁLISIS CUALITATIVO DEL PROYECTO:

1. MEJOR MODELO: {mejor_modelo}
   {'─'*76}
   El modelo {mejor_modelo} obtuvo mejor desempeño:
   • F1-Score: {metricas_rf['f1']:.4f} (Random Forest) vs {metricas_dt['f1']:.4f} (Árbol)
   • Diferencia: {diferencia_f1:.4f}

   Random Forest generalmente supera al Árbol de Decisión porque:
   - El ensemble reduce overfitting
   - Captura mejor las relaciones complejas entre features
   - Mayor robustez en generalización a datos nuevos

2. FEATURES MÁS IMPORTANTES:
   {'─'*76}
   Las features más relevantes para clasificar hongos son:

   RANDOM FOREST (Top 3):
   • {X.columns[indices_rf[0]]}: {importances_rf[indices_rf[0]]:.4f}
   • {X.columns[indices_rf[1]]}: {importances_rf[indices_rf[1]]:.4f}
   • {X.columns[indices_rf[2]]}: {importances_rf[indices_rf[2]]:.4f}

   Estas características capturan las diferencias fundamentales entre
   hongos venenosos y comestibles en el dataset.

3. LIMITACIONES DEL ENFOQUE:
   {'─'*76}
   • El dataset contiene solo características categóricas. No hay información
     numérica sobre tamaño, peso u otras medidas continuas que podrían mejorar
     la clasificación.

   • La mayoría de muestras son de ciertas especies. El desbalance de clases
     (si existe) puede favorecer la clase mayoritaria.

   • Los modelos basados en árboles pueden no capturar interacciones
     complejas entre features que requieren capas de redes neuronales.

   • No se consideraron técnicas de validación cruzada más robustas
     (k-fold cross-validation) aunque sí se usó validación estratificada.

4. MÉTRICAS FINALES EN TEST:
   {'─'*76}
   Random Forest:
   • Accuracy:  {metricas_rf['accuracy']:.4f} ({metricas_rf['accuracy']*100:.2f}%)
   • Precision: {metricas_rf['precision']:.4f} (pocos falsos positivos)
   • Recall:    {metricas_rf['recall']:.4f} (captura bien la clase mayoritaria)
   • F1-Score:  {metricas_rf['f1']:.4f}

   Árbol de Decisión:
   • Accuracy:  {metricas_dt['accuracy']:.4f} ({metricas_dt['accuracy']*100:.2f}%)
   • Precision: {metricas_dt['precision']:.4f}
   • Recall:    {metricas_dt['recall']:.4f}
   • F1-Score:  {metricas_dt['f1']:.4f}

5. RECOMENDACIONES:
   {'─'*76}
   ✓ Usar Random Forest como modelo en producción para clasificar nuevos hongos
   ✓ Considerar técnicas de data augmentation para mejorar el recall
   ✓ Recolectar más datos de hongos venenosos si es posible
   ✓ Explorar técnicas de ensemble más avanzadas (Gradient Boosting, XGBoost)
   ✓ Implementar validación cruzada k-fold para evaluación más robusta

{'='*80}
"""

print(conclusiones)

# Guardar conclusiones en un archivo de texto
with open('outputs/CONCLUSIONES.txt', 'w', encoding='utf-8') as f:
    f.write("PROYECTO FINAL: CLASIFICACIÓN DE HONGOS\n")
    f.write("="*80 + "\n")
    f.write(conclusiones)

print("✓ Conclusiones guardadas en: outputs/CONCLUSIONES.txt")

print("\n" + "="*80)
print("PROYECTO COMPLETADO EXITOSAMENTE")
print("="*80)
print("\nArchivos generados:")
print("  • outputs/01_distribucion_clases.png")
print("  • outputs/02_frecuencia_features.png")
print("  • outputs/03_comparacion_modelos.png")
print("  • outputs/04_matrices_confusion.png")
print("  • outputs/05_importancia_features.png")
print("  • outputs/CONCLUSIONES.txt")
print("\n")
