"""Convierte capturas de loopback (enlace NULL) a Ethernet para que ambas versiones de CICFlowMeter lean el mismo archivo."""
import sys
from scapy.all import Ether, IP, PcapReader, PcapWriter

src, dst = sys.argv[1], sys.argv[2]
n = 0
with PcapReader(src) as reader, PcapWriter(dst, linktype=1) as writer:
    for pkt in reader:
        if IP not in pkt:
            continue
        frame = Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02", type=0x0800) / pkt[IP]
        frame.time = pkt.time
        writer.write(frame)
        n += 1
print(dst, n, "paquetes")
