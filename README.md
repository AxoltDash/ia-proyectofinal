# Proyecto Final — Clasificación de Hongos
---
## Integrantes :
- Ramirez Luna Gibran
- Perez Servin Darshan Israel
- Morales Flores Pablo

**Aprendizaje de Máquina · Facultad de Ciencias, UNAM**

Clasificación binaria de hongos como **venenosos o comestibles** a partir de rasgos morfológicos, usando dos modelos implementados desde cero: Árbol de Decisión y Regresión Logística.

---

## Tabla de contenidos

1. [Dataset](#1-dataset)
2. [Descripción estadística](#2-descripción-estadística)
3. [Modelos elegidos y justificación](#3-modelos-elegidos-y-justificación)
4. [Régimen de entrenamiento e hiperparámetros](#4-régimen-de-entrenamiento-e-hiperparámetros)
5. [Evaluación y métricas](#5-evaluación-y-métricas)
6. [Conclusiones](#6-conclusiones)
7. [Estructura del repositorio](#7-estructura-del-repositorio)
8. [Cómo ejecutar](#8-cómo-ejecutar)

---

## 1. Dataset

| Campo | Valor |
|---|---|
| Nombre | Mushroom Classification |
| Fuente | UCI Machine Learning Repository |
| ID UCI | 73 |
| Tipo de problema | Clasificación binaria |
| Variable objetivo | `poisonous` — `e` (edible / comestible) · `p` (poisonous / venenoso) |

El dataset se descarga automáticamente al ejecutar el script mediante la librería `ucimlrepo`:

```python
from ucimlrepo import fetch_ucirepo
mushroom = fetch_ucirepo(id=73)
```

No se requiere ningún archivo CSV local.

---

## 2. Descripción estadística

### Dimensiones

| Conjunto | Muestras | Proporción |
|---|---|---|
| Total | 8 124 | 100 % |
| Entrenamiento | 5 686 | 70 % |
| Validación | 1 219 | 15 % |
| Prueba (test) | 1 219 | 15 % |

La partición es **estratificada** (misma proporción de clases en cada subconjunto).

### Rasgos

El dataset tiene **22 rasgos**, todos categóricos (nominales):

`cap-shape`, `cap-surface`, `cap-color`, `bruises`, `odor`, `gill-attachment`, `gill-spacing`, `gill-size`, `gill-color`, `stalk-shape`, `stalk-root`, `stalk-surface-above-ring`, `stalk-surface-below-ring`, `stalk-color-above-ring`, `stalk-color-below-ring`, `veil-type`, `veil-color`, `ring-number`, `ring-type`, `spore-print-color`, `population`, `habitat`.

> `stalk-root` tiene valores faltantes en el dataset original (2 480 de 8 124). Se tratan como una categoría adicional `'faltante'` para mantener todos los rasgos como cadenas homogéneas.

### Distribución de la clase objetivo

| Clase | Muestras | Porcentaje |
|---|---|---|
| Comestible (`e`) | 4 208 | 51.8 % |
| Venenoso (`p`) | 3 916 | 48.2 % |

El dataset está prácticamente balanceado.

### Descripción estadística (por rasgo)

Como todos los rasgos son categóricos, se reportan: conteo total, número de valores únicos, categoría más frecuente y su frecuencia.

| Rasgo | Total | Únicos | Más frecuente | Frecuencia |
|---|---|---|---|---|
| cap-shape | 8124 | 6 | x | 3656 |
| cap-surface | 8124 | 4 | y | 3244 |
| cap-color | 8124 | 10 | n | 2284 |
| bruises | 8124 | 2 | f | 4748 |
| odor | 8124 | 9 | n | 3528 |
| gill-attachment | 8124 | 2 | f | 7914 |
| gill-spacing | 8124 | 2 | c | 6812 |
| gill-size | 8124 | 2 | b | 5612 |
| gill-color | 8124 | 12 | b | 1728 |
| stalk-shape | 8124 | 2 | t | 4608 |
| stalk-root | 8124 | 5 | b | 3776 |
| stalk-surface-above-ring | 8124 | 4 | s | 5176 |
| stalk-surface-below-ring | 8124 | 4 | s | 4936 |
| stalk-color-above-ring | 8124 | 9 | w | 4464 |
| stalk-color-below-ring | 8124 | 9 | w | 4384 |
| veil-type | 8124 | 1 | p | 8124 |
| veil-color | 8124 | 4 | w | 7924 |
| ring-number | 8124 | 3 | o | 7488 |
| ring-type | 8124 | 5 | p | 3968 |
| spore-print-color | 8124 | 9 | w | 2388 |
| population | 8124 | 6 | v | 4040 |
| habitat | 8124 | 7 | d | 3148 |

### Visualizaciones generadas

| Archivo | Descripción |
|---|---|
| `outputs/01_distribucion_clases.png` | Distribución absoluta y porcentual de clases |
| `outputs/02_frecuencia_rasgos.png` | Frecuencia de las primeras 4 características |
| `outputs/03_comparacion_metricas.png` | Comparación lado a lado de métricas en test |
| `outputs/04_matrices_confusion.png` | Matrices de confusión de ambos modelos |
| `outputs/05_pesos_regresion.png` | Rasgos más influyentes según la regresión logística |

---

## 3. Modelos elegidos y justificación

Se implementaron **dos modelos desde cero** (sin usar las implementaciones de scikit-learn), siguiendo el material del curso:

### Árbol de Decisión (modelo no paramétrico)

Construye una jerarquía de reglas de la forma `¿rasgo == valor?`. En cada nodo selecciona el rasgo y valor que maximizan la **ganancia de información** (reducción de entropía o impureza de Gini) al bipartir el conjunto.

**Por qué es adecuado:**
- Los datos son categóricos nominales; el árbol los maneja de forma nativa, sin necesidad de codificación.
- El resultado es directamente interpretable: las ramas del árbol son las reglas de clasificación.
- No asume ninguna forma de la frontera de decisión.

### Regresión Logística (modelo paramétrico / lineal)

Estima $p(\text{venenoso} \mid x)$ mediante la función logística aplicada a una combinación lineal de los rasgos: $f(x) = \sigma(\theta \cdot x + \theta_0)$. Los pesos $\theta_i$ se aprenden con **gradiente descendiente estocástico**.

**Por qué es adecuado:**
- Sirve de contraste frente al árbol: un modelo lineal vs. uno basado en reglas.
- Los pesos $\theta_i$ dan una lectura cuantitativa de cuánto contribuye cada categoría a la predicción de "venenoso".
- Permite obtener una **probabilidad calibrada** (no solo una clase), útil cuando el costo de un error es asimétrico.

> `scikit-learn` se usa **únicamente** para `train_test_split` y el cálculo de métricas. Las implementaciones de los modelos son propias.

---

## 4. Régimen de entrenamiento e hiperparámetros

### Árbol de Decisión

| Hiperparámetro | Opciones exploradas | Seleccionado |
|---|---|---|
| Criterio de partición | `entropy` (ganancia de información), `gini` (impureza de Gini) | `entropy` |

La selección se realizó comparando la exactitud en el **conjunto de validación**. El árbol crece sin profundidad máxima hasta separar completamente el conjunto de entrenamiento.

```
criterio=entropy  -> exactitud validación = 1.0000  (~1.5 s)
criterio=gini     -> exactitud validación = 1.0000  (~1.8 s)
→ Seleccionado: entropy
```

### Regresión Logística

| Hiperparámetro | Opciones exploradas | Seleccionado |
|---|---|---|
| Tasa de aprendizaje (`lr`) | 0.01, 0.1, 0.5 | 0.01 |
| Iteraciones (`max_its`) | 50, 100 | 50 |

Búsqueda exhaustiva en grilla (6 combinaciones) evaluada en validación:

```
lr=0.01  max_its=50   -> exactitud validación = 1.0000
lr=0.01  max_its=100  -> exactitud validación = 1.0000
lr=0.10  max_its=50   -> exactitud validación = 1.0000
lr=0.10  max_its=100  -> exactitud validación = 1.0000
lr=0.50  max_its=50   -> exactitud validación = 1.0000
lr=0.50  max_its=100  -> exactitud validación = 1.0000
→ Seleccionados: lr=0.01, max_its=50  (primera combinación óptima)
```

Los rasgos categóricos se codifican con **one-hot encoding** antes de alimentar la regresión logística (116 columnas binarias), ya que asignarles enteros introduciría un orden falso que el modelo lineal interpretaría literalmente.

---

## 5. Evaluación y métricas

La evaluación final se realizó sobre el **conjunto de prueba** (1 219 muestras, nunca vistas durante el entrenamiento ni el ajuste de hiperparámetros). La clase positiva es `Venenoso`.

### Resultados

| Métrica | Árbol de Decisión | Regresión Logística |
|---|---|---|
| Exactitud (Accuracy) | **1.0000** | **1.0000** |
| Precisión (Venenoso) | **1.0000** | **1.0000** |
| Exhaustividad / Recall (Venenoso) | **1.0000** | **1.0000** |
| F1-Score | **1.0000** | **1.0000** |

### Matrices de confusión

**Árbol de Decisión**

|  | Pred. Comestible | Pred. Venenoso |
|---|---|---|
| Real Comestible | 631 | 0 |
| Real Venenoso | 0 | 588 |

**Regresión Logística**

|  | Pred. Comestible | Pred. Venenoso |
|---|---|---|
| Real Comestible | 631 | 0 |
| Real Venenoso | 0 | 588 |

Ambos modelos clasifican el conjunto de prueba sin ningún error. **0 falsos negativos** (hongos venenosos clasificados como comestibles), que es el error de mayor riesgo en este dominio.

### Rasgos más influyentes

- **Árbol de Decisión:** el primer nodo de decisión usa el rasgo `odor`, lo que indica que el olor del hongo por sí solo separa la mayor parte de las clases.
- **Regresión Logística:** las categorías con mayor peso $\theta$ hacia "venenoso" son `spore-print-color_r` (+3.575), `odor_f` (+3.215) y `odor_c` (+2.929). Las que más empujan hacia "comestible" son `odor_n` (-4.688) y `odor_l` (-2.865). Ambos modelos coinciden en que `odor` es el rasgo más discriminativo.

---

## 6. Conclusiones

### Desempeño general

Ambos modelos alcanzan **métricas perfectas (1.0)** en el conjunto de prueba. Esto no es una anomalía: el dataset de hongos UCI es conocido por ser casi perfectamente separable a partir de sus rasgos morfológicos. La conclusión relevante no es "cuál modelo gana por unas décimas", sino que el problema es **estructuralmente fácil** y que el verdadero contraste entre modelos está en su interpretabilidad.

### Qué aprendió cada modelo

El árbol de decisión expresa su conocimiento como una cadena de reglas legibles (`¿odor == 'f'? → Venenoso`), directamente verificables por un experto. La regresión logística lo expresa como pesos numéricos que cuantifican la contribución de cada categoría. Ambos enfoques coinciden en señalar `odor` como el rasgo más discriminativo.

### La métrica crítica en este dominio

En clasificación de hongos el **recall de la clase "venenoso"** es la métrica prioritaria: un falso negativo (clasificar un hongo venenoso como comestible) tiene consecuencias graves. Ambos modelos logran recall = 1.0, lo que los hace aptos para este uso.

### Limitaciones

- El dataset contiene únicamente rasgos categóricos morfológicos; variables continuas (tamaño, peso) podrían ser necesarias en escenarios más difíciles.
- La alta separabilidad hace que las métricas sean optimistas respecto a datos del mundo real.
- El árbol crece sin poda ni profundidad máxima; con datos ruidosos tendería al sobreajuste.
- La regresión logística usa gradiente descendiente básico; sobre datos linealmente separables los pesos pueden crecer sin cota teórica (aquí se controla con `max_its` fijo).
- Se usó una única partición fija; una validación cruzada k-fold daría una estimación más robusta de la generalización.

### Recomendaciones

- Para uso en campo se preferiría el **árbol de decisión**: sus reglas son auditables y no requieren conocimientos de álgebra lineal para ser verificadas.
- Si se necesita una **probabilidad calibrada** (p.ej. para definir un umbral de alerta), la regresión logística es más adecuada.
- Ante un costo asimétrico de errores, se puede subir el umbral de decisión de la regresión logística para favorecer el recall de "venenoso".

---

## 7. Estructura del repositorio

```
proyecto-final/
├── proyecto_final.py      # Script principal (corre de principio a fin)
├── INSTRUCTIONS.md        # Rúbrica oficial del proyecto (pero en md jeje)
├── README.md              # Este archivo
└── outputs/               # Generado automáticamente al ejecutar el script
    ├── 01_distribucion_clases.png
    ├── 02_frecuencia_rasgos.png
    ├── 03_comparacion_metricas.png
    ├── 04_matrices_confusion.png
    ├── 05_pesos_regresion.png
    ├── arbol_decision.txt     # Árbol completo en texto legible
    └── CONCLUSIONES.txt       # Conclusiones exportadas
```

---

## 8. Cómo ejecutar

### Requisitos

- Python 3.9+
- El directorio incluye un entorno virtual (`venv/`) con todas las dependencias instaladas.

### Ejecución

```bash
# 1. Activar el entorno virtual
source venv/bin/activate

# 2. Ejecutar el script (descarga el dataset y genera todos los outputs)
python proyecto_final.py

# 3. Desactivar al terminar
deactivate
```

El script corre de principio a fin sin intervención manual. Todos los resultados (métricas en consola + gráficas en `outputs/`) se generan automáticamente.

### Dependencias

| Paquete | Versión | Uso |
|---|---|---|
| `numpy` | 2.4.6 | Álgebra lineal |
| `pandas` | 3.0.3 | Manipulación del dataset |
| `matplotlib` | 3.10.9 | Gráficas |
| `seaborn` | 0.13.2 | Heatmaps de matrices de confusión |
| `scikit-learn` | 1.8.0 | `train_test_split` y métricas |
| `ucimlrepo` | 0.0.7 | Descarga del dataset UCI |

Para instalar desde cero:

```bash
pip install numpy==2.4.6 pandas==3.0.3 matplotlib==3.10.9 seaborn==0.13.2 scikit-learn==1.8.0 ucimlrepo==0.0.7
```

---

*Facultad de Ciencias, UNAM — Aprendizaje de Máquina*
