# Detector DDoS RF 11 con victima HTTP

Sistema de deteccion DoS/DDoS para pruebas reales en macOS. Esta version usa un modelo Random Forest multiclase de 11 caracteristicas y levanta dos servicios separados:

- `4100`: dashboard/API del detector.
- `8080`: servidor victima desde la carpeta `victima/`.

## Modelo activo

- Tipo cargado por la app: `MODEL_KIND=xgboost`
- Modelo real: `RandomForestClassifier`
- Carpeta: `modelo_rf_multiclase`
- Variables: 11
- Clases: `Benign`, `DNS`, `LDAP`, `MSSQL`, `NTP`, `NetBIOS`, `Portmap`, `SNMP`, `Syn`, `TFTP`, `UDP`, `UDPLag`, `WebDDoS`

El nombre `xgboost.pkl` se mantiene solo por compatibilidad con `app.py`.

## Estructura principal

```text
.
├── app.py                    # API FastAPI, dashboard y motor de deteccion
├── web/                      # Dashboard del detector
├── victima/                  # Pagina/servidor victima para el puerto 8080
├── modelo_rf_multiclase/     # Artefactos del Random Forest de 11 variables
├── scripts/
│   ├── iniciar_admin_mac_rf11.sh
│   ├── iniciar_local_rf11.sh
│   ├── apagar_rf11.sh
│   └── convertir_pcap.py
└── requirements_rf11.txt
```

## Arranque con captura real en macOS

```bash
cd '/ruta/al/proyecto'
./scripts/instalar_entorno.sh
sudo CAPTURE_INTERFACE=en0 ./scripts/iniciar_admin_mac_rf11.sh
```

Abre el detector:

```text
http://127.0.0.1:4100
```

La victima queda en:

```text
http://127.0.0.1:8080
```

Por defecto, los PCAP y CSV de captura se eliminan automaticamente despues de procesarse. Si necesitas conservarlos para depuracion:

```bash
sudo KEEP_CAPTURE_FILES=true ./scripts/iniciar_admin_mac_rf11.sh
```

## Arranque local sin captura

```bash
cd '/ruta/al/proyecto'
./scripts/iniciar_local_rf11.sh
```

## Prueba desde Windows sin ApacheBench

Cambia `IP_DEL_MAC` por la IP real del equipo macOS:

```powershell
$u="http://IP_DEL_MAC:8080/"; 1..10000 | ForEach-Object -Parallel { Invoke-WebRequest -Uri $using:u -UseBasicParsing | Out-Null } -ThrottleLimit 200
```

## Lista blanca

La lista blanca se configura con `WHITELIST_IPS`. Por defecto:

```bash
WHITELIST_IPS="127.0.0.1"
```

Ejemplo:

```bash
sudo WHITELIST_IPS="127.0.0.1,172.64.148.235,17.248.201.71" CAPTURE_INTERFACE=en0 ./scripts/iniciar_admin_mac_rf11.sh
```

## Apagar

```bash
sudo ./scripts/apagar_rf11.sh
```
