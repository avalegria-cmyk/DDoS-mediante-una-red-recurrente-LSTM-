#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-eth0}"
PCAP_DIR="${PCAP_DIR:-/opt/ddos-shield/data/pcap}"
DURATION="${DURATION:-15}"

mkdir -p "$PCAP_DIR"
OUT="$PCAP_DIR/captura-$(date +%Y%m%d-%H%M%S).pcap"

echo "Capturando $DURATION segundos en interfaz $IFACE..."
/usr/bin/timeout "$DURATION" /usr/bin/tcpdump -i "$IFACE" -nn -w "$OUT"
echo "$OUT"
