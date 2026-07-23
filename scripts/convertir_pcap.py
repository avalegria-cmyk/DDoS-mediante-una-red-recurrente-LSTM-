from __future__ import annotations

import sys
from pathlib import Path

from cicflowmeter.flow_session import FlowSession
from scapy.utils import PcapReader


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Uso: convertir_pcap.py ENTRADA.pcap SALIDA.csv")

    input_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el PCAP: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = FlowSession(
        output_mode="csv",
        output=str(output_path),
        fields=None,
        verbose=False,
    )
    try:
        with PcapReader(str(input_path)) as packets:
            for packet in packets:
                session.process(packet)
    finally:
        session.flush_flows()

    if not output_path.exists():
        raise RuntimeError("CICFlowMeter no genero el archivo CSV")
    print(output_path)


if __name__ == "__main__":
    main()
