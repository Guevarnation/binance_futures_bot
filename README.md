# Bot de Trading de Futuros de Binance con Modelo de Machine Learning (Random Forest)

Este proyecto implementa un bot de trading automatizado para futuros de Binance (específicamente BTCUSDT en la Testnet por defecto), utilizando un modelo de Machine Learning (Random Forest) para tomar decisiones de compra/venta/mantener.

## Descripción General

El bot utiliza datos históricos y en tiempo real para alimentar un modelo de Random Forest pre-entrenado. Las características (features) utilizadas para entrenar y ejecutar el modelo incluyen:

- **Indicadores Técnicos:**
  - RSI (Relative Strength Index)
  - Diferencia entre Medias Móviles Simples (SMA Corta - SMA Larga)
  - MACD (Línea, Histograma, Señal)
  - ATR (Average True Range) - Volatilidad
  - Cambio Porcentual del Volumen
- **Indicador de Sentimiento:**
  - Fear & Greed Index (obtenido de [alternative.me](https://alternative.me/crypto/fear-and-greed-index/))
- **Características Desfasadas (Lagged Features):**
  - Valores anteriores (últimos 3 periodos) de RSI, Diferencia SMA, Histograma MACD, ATR y Cambio de Volumen para dar contexto histórico al modelo.

## Configuración del Proyecto

Sigue estos pasos para configurar y ejecutar el bot:

1.  **Clonar el Repositorio:**

    ```bash
    git clone <url-del-repositorio>
    cd <directorio-del-repositorio>
    ```

2.  **Crear Entorno Virtual:** (Recomendado)

    ```bash
    python -m venv venv
    # En macOS/Linux
    source venv/bin/activate
    # En Windows
    # venv\Scripts\activate
    ```

3.  **Instalar Dependencias:**
    Asegúrate de que todas las librerías necesarias estén listadas en `requirements.txt`. Luego, instala:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Crear Archivo `.env`:**
    Crea un archivo llamado `.env` en la raíz del proyecto. Este archivo almacenará tus claves de API de Binance y el tamaño deseado para las operaciones. **¡IMPORTANTE!** Para empezar, usa claves generadas en la [Binance Testnet](https://testnet.binancefuture.com/) para no arriesgar dinero real.
    ```dotenv
    BINANCE_API_KEY=TU_API_KEY_DE_TESTNET_AQUI
    BINANCE_SECRET_KEY=TU_SECRET_KEY_DE_TESTNET_AQUI
    TRADE_NOTIONAL=200 # Valor nocional deseado por operación en USDT (ej. 200)
    ```
    _(El bot usará 200 USDT como valor por defecto si `TRADE_NOTIONAL` no se encuentra o no es válido)._

## Entrenamiento del Modelo (`train_model.py`)

Antes de poder ejecutar el bot con la estrategia `proyecto`, necesitas entrenar el modelo de Random Forest.

1.  **Propósito:** El script `train_model.py` se encarga de:

    - Descargar datos históricos de precios (Klines) de Binance para BTCUSDT (intervalo de 15m por defecto).
    - Descargar datos históricos del índice Fear & Greed.
    - Calcular todas las características (indicadores técnicos, lags).
    - Crear una variable objetivo (`target`). Actualmente, define:
      - `1` (Comprar): Si el precio de cierre `10` periodos en el futuro es > 0.2% más alto que el actual.
      - `-1` (Vender): Si el precio de cierre `10` periodos en el futuro es < 0.2% más bajo que el actual.
      - `0` (Mantener): En los demás casos.
      - _(Puedes experimentar ajustando `TARGET_LOOKAHEAD` y `TARGET_THRESHOLD` en el script)._
    - Dividir los datos en conjuntos de entrenamiento y prueba.
    - **Escalar las características:** Utiliza `StandardScaler` para normalizar los datos, lo cual puede ayudar al rendimiento del modelo. El scaler entrenado se guarda en `feature_scaler.joblib`.
    - **Optimizar Hiperparámetros:** Usa `RandomizedSearchCV` para probar diferentes combinaciones de parámetros del `RandomForestClassifier` y encontrar la mejor configuración basada en la precisión durante la validación cruzada.
    - Entrenar el modelo final con los mejores parámetros encontrados.
    - Evaluar el modelo en el conjunto de prueba (accuracy, classification report, confusion matrix).
    - **Guardar el Modelo:** Si la precisión del modelo entrenado en el conjunto de prueba supera el umbral del 52% (definido en el script), guarda el modelo como `proyecto_model.joblib`. Si no supera el umbral, no se guarda y deberás ajustar características, el objetivo o los parámetros de entrenamiento.

2.  **Ejecutar Entrenamiento:**
    ```bash
    python train_model.py
    ```
    Este proceso puede tardar varios minutos debido a la optimización de hiperparámetros (`RandomizedSearchCV`). Revisa los logs en la terminal y en `logs/training.log`.

## Proceso de Desarrollo del Modelo (Random Forest)

Esta sección detalla los pasos seguidos para construir y evaluar el modelo de Random Forest utilizado en la estrategia `proyecto`, alineándose con las fases requeridas en el proyecto integrador.

### 1. Definición y Planificación

Se seleccionó la pista de **"Bot de trading automático"** como enfoque principal. El objetivo fue entrenar un modelo de clasificación (Random Forest) para predecir señales de trading (Comprar/Mantener/Vender) para el par BTCUSDT en un intervalo de 15 minutos, utilizando datos históricos de Binance y el índice "Fear & Greed". Se eligió Random Forest como primer modelo a implementar por su robustez, buen rendimiento general en problemas de clasificación y su capacidad para manejar features diversas y no lineales, además de ser uno de los modelos explícitamente sugeridos.

### 2. Recolección y Preparación de Datos

- **Datos de Mercado:** Se utilizó la librería `python-binance` para descargar datos históricos de Klines (velas) para BTCUSDT con intervalo de 15 minutos desde Binance, abarcando desde el 1 de enero de 2022 hasta la fecha actual (configurable en `train_model.py`). Estos datos incluyen precios de apertura, máximo, mínimo, cierre y volumen.
- **Índice Fear & Greed (F&G):** Se obtuvo el historial del índice desde la API pública de `alternative.me`. Este índice se considera una feature de sentimiento del mercado.
- **Limpieza y Fusión:**
  - Los datos de Klines se limpiaron convirtiendo columnas a tipos numéricos.
  - Los datos de F&G (diarios) se fusionaron con los datos de Klines (15 minutos) utilizando la fecha. Se aplicó `forward fill` (`ffill()`) para asignar el valor de F&G más reciente a cada vela de 15 minutos dentro del mismo día.
  - Se eliminaron filas iniciales con valores `NaN` resultantes del cálculo de indicadores técnicos o lags.

### 3. Ingeniería de Características (Feature Engineering)

Se crearon diversas características técnicas y de sentimiento para alimentar al modelo, basadas en prácticas comunes de análisis técnico:

- **Indicadores Técnicos:**
  - `RSI (14)`: Mide la magnitud de los cambios recientes de precios para evaluar condiciones de sobrecompra o sobreventa.
  - `SMA_diff`: Diferencia entre la media móvil simple corta (20 periodos) y larga (50 periodos), indicando tendencias a corto vs largo plazo.
  - `MACD (12, 26, 9)`: Indicador de momento que sigue tendencias (línea MACD, línea de señal, histograma).
  - `ATR (14)`: Mide la volatilidad del mercado.
  - `volume_change`: Cambio porcentual en el volumen, indicando la fuerza de un movimiento de precio.
- **Sentimiento:**
  - `fear_and_greed`: Valor numérico del índice F&G.
- **Características Desfasadas (Lags):** Para dar contexto histórico al modelo, se añadieron los valores de `RSI`, `SMA_diff`, `MACD Histogram`, `ATR` y `volume_change` de los 1, 2 y 3 periodos anteriores.

### 4. Definición de la Variable Objetivo (`target`)

Se optó por un enfoque de clasificación con tres clases (Comprar=1, Mantener=0, Vender=-1). La señal se determinó comparando el precio de cierre actual con el precio `TARGET_LOOKAHEAD` (10 periodos de 15min) en el futuro. Si el retorno futuro superaba un `TARGET_THRESHOLD` (0.2%), se asignaba 1 (Comprar); si era menor que -0.2%, se asignaba -1 (Vender); de lo contrario, 0 (Mantener). Estos parámetros se ajustaron experimentalmente para buscar un balance entre las clases.

### 5. Desarrollo del Modelo (Random Forest)

- **Escalado de Características:** Se aplicó `StandardScaler` de `scikit-learn` a todas las features antes del entrenamiento. Esto normaliza los datos (media 0, desviación estándar 1), lo cual es crucial para muchos algoritmos de ML y puede mejorar el rendimiento y la convergencia, especialmente si se usaran otros modelos como Redes Neuronales o KNN posteriormente. El scaler entrenado se guarda (`feature_scaler.joblib`) para usarlo consistentemente en las predicciones en vivo.
- **Ajuste de Hiperparámetros:** Se utilizó `RandomizedSearchCV` de `scikit-learn`. Esta técnica prueba combinaciones aleatorias de hiperparámetros (como `n_estimators`, `max_depth`, `min_samples_split`, etc.) dentro de rangos definidos y utiliza validación cruzada (3 folds) para encontrar la combinación que maximiza la métrica de evaluación seleccionada (en este caso, `accuracy`). Es más eficiente que `GridSearchCV` cuando el espacio de búsqueda es grande. Se usó `class_weight='balanced'` para mitigar el impacto de clases desbalanceadas.
- **Entrenamiento Final:** El modelo final se entrenó utilizando los mejores hiperparámetros encontrados por `RandomizedSearchCV` sobre todo el conjunto de datos de entrenamiento escalado.

### 6. Evaluación Cuantitativa

El rendimiento del modelo entrenado se evaluó en el conjunto de prueba (datos no vistos durante el entrenamiento) utilizando las siguientes métricas:

- **Accuracy:** Porcentaje total de predicciones correctas.
- **Classification Report:** Incluye:
  - **Precision:** De todas las veces que el modelo predijo una clase, ¿cuántas veces acertó? (TP / (TP + FP))
  - **Recall:** De todas las instancias reales de una clase, ¿cuántas identificó correctamente el modelo? (TP / (TP + FN))
  - **F1-Score:** Media armónica de Precision y Recall, útil para clases desbalanceadas.
- **Confusion Matrix:** Tabla que visualiza el rendimiento, mostrando cuántas veces se predijo correctamente cada clase y cómo se confundieron las clases incorrectas (verdaderos positivos, falsos positivos, verdaderos negativos, falsos negativos).
- **Simulated ROI & Sharpe Ratio:** Se calculó un retorno simulado aplicando las señales de compra/venta del modelo al retorno real futuro en el conjunto de prueba. A partir de estos retornos simulados, se calculó el ROI total acumulado (compuesto) y el Sharpe Ratio anualizado (asumiendo tasa libre de riesgo de 0) para estimar la rentabilidad ajustada al riesgo de la estrategia modelada.
- **Feature Importance:** Se extrajo la importancia de cada característica según el modelo Random Forest, indicando qué features contribuyeron más a las decisiones del modelo.

El modelo entrenado (`proyecto_model.joblib`) y el scaler (`feature_scaler.joblib`) solo se guardan si la precisión en el conjunto de prueba supera un umbral predefinido (actualmente 52%), asegurando un mínimo de calidad antes de su uso por el bot.

## Interpretación de los Resultados del Entrenamiento

Después de ejecutar `python train_model.py`, obtendrás una salida detallada en la terminal. Aquí te explicamos qué significan las partes más importantes, usando los resultados de ejemplo proporcionados:

1.  **Inicio y Carga de Datos:**

    - El script inicia y confirma la descarga de datos históricos de Klines (velas de 15 min para BTCUSDT) y del índice Fear & Greed.
    - Informa cuántos registros se obtuvieron (ej. `Fetched 116732 klines`, `Fetched 2647 F&G records`).

2.  **Cálculo de Features y Target:**

    - Confirma el cálculo de las 23 características (RSI, MACD, ATR, SMAs, F&G, Lags, etc.).
    - Muestra la distribución de la variable objetivo (`target`) después de su creación (ej. `-1: 33.9%, 0: 33.8%, 1: 32.3%`), indicando qué tan balanceadas quedaron las clases de Vender/Mantener/Comprar antes de entrenar.

3.  **Preparación y División de Datos:**

    - Indica cuántas filas se eliminaron por tener valores `NaN` (ej. `Dropped 52 rows`). Esto es normal al inicio por el cálculo de indicadores y lags.
    - Informa el tamaño de los conjuntos de entrenamiento y prueba (ej. `Training set size: 93344, Test set size: 23336`).
    - Confirma que las características fueron escaladas con `StandardScaler` y que el scaler se guardó (ej. `Scaler saved to feature_scaler.joblib`).

4.  **Optimización de Hiperparámetros (`RandomizedSearchCV`):**

    - La línea `Fitting 3 folds for each of 50 candidates, totalling 150 fits` es clave:
      - **`50 candidates`**: Se probaron 50 combinaciones aleatorias de hiperparámetros (la "receta" del modelo).
      - **`3 folds`**: Para cada combinación, se usó validación cruzada de 3 pasos (se divide el set de entrenamiento en 3, se entrena con 2 y se valida con 1, rotando). Esto ayuda a obtener una evaluación más robusta de qué tan buena es cada receta.
      - **`150 fits`**: En total, se realizaron 150 pequeños entrenamientos/validaciones internas para encontrar la mejor combinación.
    - `Best parameters found`: Muestra la combinación ganadora de hiperparámetros (ej. `{'max_depth': 50, ... 'n_estimators': 317}`).
    - `Best cross-validation accuracy score`: Indica la precisión promedio (ej. `0.5122` o 51.2%) que obtuvo la mejor receta _durante la validación cruzada_ con los datos de entrenamiento.

5.  **Evaluación Final del Modelo (en el Conjunto de Prueba):**

    - `Test Set Accuracy (Tuned Model)`: **¡La métrica más importante!** Muestra la precisión del modelo final con los mejores parámetros al ser probado en datos _totalmente nuevos_ que no usó para entrenar (ej. `0.5558` o 55.6%). Este valor decide si el modelo se guarda o no.
    - `Classification Report`: Desglosa el rendimiento por clase:
      - `precision`: Qué tan preciso fue el modelo cuando _predijo_ una clase específica.
      - `recall`: Qué porcentaje de las instancias _reales_ de una clase logró identificar el modelo.
      - `f1-score`: Una media entre precisión y recall, útil para comparar el rendimiento general por clase.
      - `support`: Cuántas instancias de cada clase había en el conjunto de prueba.
    - `Confusion Matrix`: Una tabla que muestra exactamente cuántas predicciones fueron correctas (diagonal principal) y cómo se confundieron las incorrectas (fuera de la diagonal).
    - `Feature Importances`: Lista las características que más influyeron en las predicciones del modelo (ej. `atr`, `fear_and_greed`, `volume_change_lag_3` fueron las más importantes en el ejemplo).

6.  **Guardado del Modelo:**
    - Si la `Test Set Accuracy` supera el umbral (52%), confirma que el modelo y el scaler se guardaron correctamente (ej. `Model and scaler saved successfully.`). Si no, mostrará una advertencia indicando que no se guardó.

En resumen, esta salida te permite entender la calidad de los datos usados, cómo se encontró la mejor configuración para el modelo Random Forest, y qué tan bien se espera que funcione ese modelo final al predecir señales de trading en datos nuevos, basándose en métricas estándar de Machine Learning.

## Ejecución del Bot (`main.py`)

Una vez que el entrenamiento haya finalizado y los archivos `proyecto_model.joblib` y `feature_scaler.joblib` se hayan guardado:

1.  **Propósito:** El script `main.py` ejecuta el ciclo principal del bot:

    - Carga las claves de API y el `TRADE_NOTIONAL` desde `.env`.
    - Se conecta a Binance Futures.
    - Instancia la estrategia seleccionada (en este caso, `ProyectoStrategy`), pasándole el valor nocional configurado.
      - `ProyectoStrategy` carga el modelo `proyecto_model.joblib` y el scaler `feature_scaler.joblib`.
    - Entra en un bucle infinito (a menos que se use `--once`):
      - Obtiene los datos de mercado más recientes.
      - Calcula las características necesarias.
      - **Escala las características** usando el `feature_scaler.joblib` guardado.
      - Usa el modelo cargado (`proyecto_model.joblib`) para predecir la acción (1, -1, o 0).
      - Ejecuta órdenes de compra/venta en Binance Testnet si la señal es 1 o -1 y no hay una posición abierta en la misma dirección. El tamaño de la orden se calcula para que su valor nocional sea aproximadamente igual al `TRADE_NOTIONAL` configurado.
      - Espera hasta la siguiente vela (según el intervalo, ej. 15m).

2.  **Ejecutar el Bot:**
    Para correr la estrategia específica de Machine Learning:

    ```bash
    python main.py --strategy proyecto --interval 15m
    ```

    _(Puedes cambiar el `--interval` si entrenaste el modelo para otro intervalo, aunque el código actual asume 15m para la estrategia `proyecto`)._

3.  **Detener el Bot:** Presiona `Ctrl + C` en la terminal.

## Disclaimer / Advertencia

- **Riesgo:** El trading de criptomonedas, especialmente futuros, conlleva un alto riesgo. Puedes perder tu inversión rápidamente.
- **No es Asesoramiento Financiero:** Este código es solo para fines educativos y de demostración. No constituye asesoramiento financiero.
- **Testnet:** **UTILIZA SIEMPRE LA TESTNET DE BINANCE** mientras desarrollas y pruebas para evitar pérdidas reales.
- **Sin Garantías:** No hay garantía de que este bot genere ganancias. El rendimiento pasado no asegura resultados futuros. Los mercados cambian y los modelos pueden dejar de ser efectivos.
- **Mejoras Necesarias:** La lógica actual de gestión de órdenes (tamaño de posición, take profit, stop loss) es muy básica y necesita ser desarrollada robustamente antes de considerar cualquier uso con fondos reales (lo cual no se recomienda sin una comprensión profunda de los riesgos y una validación exhaustiva).

Usa este proyecto de forma responsable y bajo tu propio riesgo.

## Features

- Connect to Binance Futures Testnet API
- Execute trades based on customizable strategies
- Monitor account balance and positions
- Track and analyze trading history

## Files Structure

- `main.py`: Entry point for the trading bot
- `binance_client.py`: Handles Binance API connection and operations
- `strategies.py`: Defines trading strategies
- `utils.py`: Utility functions for data processing and analysis
- `.env.example`: Example environment variables file

# binance_futures_bot
