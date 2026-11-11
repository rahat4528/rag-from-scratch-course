# Novasphere OS — Technical Product FAQ (2026)

## Service Level Agreements (SLA) & Reliability

### What uptime SLA is guaranteed for the Enterprise Tier?
The Enterprise Tier guarantees a **99.995%** service uptime SLA, backed by multi-region active-active coordinator clusters. Professional Tier guarantees 99.95%, and Starter Tier provides best-effort 99.5% availability.

### How does ZeroGravity Failover work, and which plan includes it?
ZeroGravity Automatic Failover is exclusively included in the **Enterprise Plan** ($4,200/month). If a primary edge gateway or cluster coordinator becomes unresponsive for more than **40 milliseconds**, ZeroGravity automatically transfers fleet state ownership to a standby peer node without dropping active safety heartbeats.

---

## Edge Compatibility & ROS2 Support

### Which ROS2 distributions are natively supported by Novasphere Edge Agent?
Novasphere Edge Agent v4+ natively supports three long-term/active ROS2 distributions on Ubuntu 22.04 and 24.04 ARM64/x86_64 targets:
- **Humble Hawksbill**
- **Iron Irwini**
- **Jazzy Jalisco**

Older ROS1 distributions (such as Noetic) require the legacy ROS-Bridge adapter container and are not natively supported.

---

## Troubleshooting & Error Codes

### How do I resolve error code `ERR_QUANTUM_TIMEOUT`?
`ERR_QUANTUM_TIMEOUT` occurs when the edge agent cannot complete its high-frequency QuantumCache v4 state handshake with the local coordinator. Follow these three steps in order:
1. Verify that host and perimeter firewalls permit inbound and outbound UDP traffic on **port 9872**.
2. Check hardware time synchronization and ensure **NTP/PTP clock drift is less than 5 milliseconds (< 5 ms)** across all nodes in the fleet.
3. Restart the local edge daemon by running:
   ```bash
   sudo systemctl restart novasphere-edge
   ```

### What encryption and offline buffering limits apply to edge nodes?
Each edge agent buffers up to 4 GB of compressed Parquet telemetry locally during WAN disconnects and synchronizes using TLS 1.3 mutual authentication upon reconnect.

