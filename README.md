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
    Crea un archivo llamado `.env` en la raíz del proyecto. Este archivo almacenará tus claves de API de Binance. **¡IMPORTANTE!** Para empezar, usa claves generadas en la [Binance Testnet](https://testnet.binancefuture.com/) para no arriesgar dinero real.
    ```dotenv
    BINANCE_API_KEY=TU_API_KEY_DE_TESTNET_AQUI
    BINANCE_SECRET_KEY=TU_SECRET_KEY_DE_TESTNET_AQUI
    ```

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

## Ejecución del Bot (`main.py`)

Una vez que el entrenamiento haya finalizado y los archivos `proyecto_model.joblib` y `feature_scaler.joblib` se hayan guardado:

1.  **Propósito:** El script `main.py` ejecuta el ciclo principal del bot:

    - Carga las claves de API desde `.env`.
    - Se conecta a Binance Futures.
    - Instancia la estrategia seleccionada (en este caso, `ProyectoStrategy`).
      - `ProyectoStrategy` carga el modelo `proyecto_model.joblib` y el scaler `feature_scaler.joblib`.
    - Entra en un bucle infinito (a menos que se use `--once`):
      - Obtiene los datos de mercado más recientes.
      - Calcula las características necesarias.
      - **Escala las características** usando el `feature_scaler.joblib` guardado.
      - Usa el modelo cargado (`proyecto_model.joblib`) para predecir la acción (1, -1, o 0).
      - Ejecuta órdenes de compra/venta en Binance Testnet si la señal es 1 o -1 y no hay una posición abierta en la misma dirección (la lógica de tamaño de orden y TP/SL es básica y debe mejorarse).
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
