# DDoS Mediante una Red Recurrente LSTM

Sistema de deteccion DoS/DDoS para macOS basado en un modelo LSTM reentrenado con variables que pueden obtenerse desde trafico real. Esta version no depende del modelo original de 77 caracteristicas del dataset; usa 47 caracteristicas operacionalmente disponibles y una capa complementaria por tasa para pruebas reales con trafico HTTP.

## Contenido

```text
.
├── app.py                         # API FastAPI, dashboard y motor de deteccion
├── modelo/                        # Modelo LSTM final usado por el sistema
├── modelos_lstm_reales_47/        # Copia de artefactos del entrenamiento
├── scripts/
│   ├── iniciar_admin_mac_lstm.sh          # Captura real por red, interfaz en0
│   ├── iniciar_admin_mac_lstm_local.sh    # Prueba local, interfaz lo0
│   ├── convertir_pcap.py                  # Conversor PCAP -> CSV de flujos
│   └── probar_pipeline_lstm_reales.py     # Prueba rapida del pipeline
├── web/                           # Interfaz del dashboard
├── data/                          # PCAPs, CSVs y estado generado en ejecucion
└── requirements.txt
```

## Modelo activo

- Arquitectura: LSTM.
- Archivo del modelo: `modelo/modelo_lstm_ddos.keras`.
- Scaler: `modelo/scaler_lstm_gru.pkl`.
- Variables: `modelo/feature_cols_lstm_gru.json`.
- Metadata: `modelo/info_experimento_lstm_gru.json`.
- Entrada esperada: ventanas de `10` flujos con `47` caracteristicas.
- Umbral de IA: `0.65`.

La columna `IA` del dashboard muestra la probabilidad directa del LSTM. Si aparece `0% IA` pero el evento dice `DDoS/DoS por tasa`, significa que el ataque fue marcado por la regla complementaria de trafico excesivo contra el puerto protegido.

## Por que 47 variables

El dataset CIC-DDoS2019 trae 77 caracteristicas, pero en trafico real no siempre se pueden obtener todas de forma consistente. Este sistema usa variables calculables desde paquetes capturados: protocolo, duracion, conteos de paquetes, bytes, tasas, tiempos entre paquetes, tamanos y banderas disponibles.

Resultado comparativo del entrenamiento:

- LSTM original con 77 variables: F1 `0.981575`.
- LSTM con 47 variables reales: F1 `0.980815`.
- LSTM con 20 variables seleccionadas: F1 `0.959859`.

La version de 47 variables mantiene el enfoque LSTM de la investigacion y conserva un rendimiento muy cercano al modelo de 77 variables.

## Requisitos

- macOS.
- Python 3.12.
- Permisos de administrador para capturar paquetes con `tcpdump`.
- Servidor victima escuchando en el puerto `8080`.

Crear el entorno virtual local e instalar dependencias:

```bash
cd "/Users/adonisalegria/Documents/Universidad/Seguridad/ENTREGA V2"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Los scripts tambien son compatibles con el entorno antiguo `../implementacion/.venv` si existe, pero para Git se recomienda usar `.venv` dentro del repositorio.

## Prueba rapida del modelo

Esta prueba no captura trafico. Solo valida que el modelo, scaler y columnas carguen correctamente.

```bash
cd "/Users/adonisalegria/Documents/Universidad/Seguridad/ENTREGA V2"

.venv/bin/python scripts/probar_pipeline_lstm_reales.py
```

Salida esperada parecida:

```text
OK LSTM real: 10 flujos, 47 variables, probabilidad ...
```

## Ejecutar el dashboard

El dashboard queda disponible en:

```text
http://127.0.0.1:4100
```

### Modo local: ataque desde el mismo Mac

Usa este modo cuando el atacante y la victima estan en la misma Mac. Captura por `lo0`.

```bash
cd "/Users/adonisalegria/Documents/Universidad/Seguridad/ENTREGA V2"

sudo scripts/iniciar_admin_mac_lstm_local.sh
```

Luego abre el dashboard y pulsa **Iniciar monitoreo**.

Ejemplo de ataque local con ApacheBench:

```bash
ab -n 5000 -c 100 http://127.0.0.1:8080/api/ping
```

En este modo se usa `RATE_ATTACK_PPS=500` para que la prueba local con `ab` se marque claramente como `DDoS/DoS por tasa`.

### Modo red: ataque desde otra maquina

Usa este modo cuando otra computadora de la red ataca la IP del Mac. Captura por `en0`.

```bash
cd "/Users/adonisalegria/Documents/Universidad/Seguridad/ENTREGA V2"

sudo scripts/iniciar_admin_mac_lstm.sh
```

Luego, desde la otra maquina:

```bash
ab -n 5000 -c 100 http://IP_DEL_MAC:8080/api/ping
```

Ejemplo si el Mac tiene IP `192.168.0.105`:

```bash
ab -n 5000 -c 100 http://192.168.0.105:8080/api/ping
```

## Apagar el sistema

Buscar procesos activos:

```bash
ps aux | grep -E "uvicorn app:app|tcpdump|MODELO LSTM"
```

Apagar detector y captura:

```bash
sudo kill PID_DETECTOR PID_TCPDUMP
```

Si tambien esta encendido el servidor victima:

```bash
kill PID_VICTIMA
```

## Variables de configuracion

Estas variables pueden ajustarse antes de ejecutar el script:

```bash
export PREDICTION_THRESHOLD="0.65"   # Umbral del LSTM
export RATE_ATTACK_PPS="1000"        # Umbral por tasa en modo red
export PROTECTED_PORTS="80,443,8080" # Puertos protegidos
export AUTO_BLOCK="false"            # Detectar sin bloquear firewall
export CAPTURE_SECONDS="15"          # Duracion de cada captura
```

En modo local, `scripts/iniciar_admin_mac_lstm_local.sh` fuerza:

```bash
CAPTURE_INTERFACE="lo0"
RATE_ATTACK_PPS="500"
```

En modo red, `scripts/iniciar_admin_mac_lstm.sh` usa:

```bash
CAPTURE_INTERFACE="en0"
AUTO_BLOCK="false"
```

## Notas importantes

- `hping3` no viene instalado por defecto en macOS. Para la demo principal se recomienda `ab` porque ataca directamente el servidor victima HTTP en `8080`.
- Si el ataque se lanza desde el mismo Mac, se debe usar `lo0`; si se lanza desde otra maquina, se debe usar `en0`.
- Un PCAP vacio en `lo0` puede ocurrir cuando no hubo trafico durante una ventana de captura. El sistema lo ignora para no detener el monitoreo.
- `127.0.0.1` esta en whitelist para evitar bloqueo local accidental. En pruebas locales puede detectarse DDoS, pero no necesariamente aparecer como IP bloqueada.
- `AUTO_BLOCK=false` significa que el sistema detecta y registra eventos, pero no modifica el firewall.

## Frase sugerida para documentacion academica

Se reentreno el modelo LSTM con 47 caracteristicas operacionalmente disponibles en trafico real, eliminando dependencia de campos que no siempre se obtienen de forma consistente. El sistema conserva la entrada temporal de 10 flujos y combina la probabilidad del LSTM con una regla operacional por tasa para detectar rafagas HTTP reales contra puertos protegidos.
