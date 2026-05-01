# Dual-Port KV Event Patch — Build Instructions

## What This Patch Does

Removes the `ConnectionMode::Grpc` gate from SMG's `KvEventMonitor`, allowing
HTTP-mode workers to also receive KV cache event subscriptions via a separate
gRPC port (bridge sidecar).

This enables event-driven cache-aware routing in setups where:
- Inference uses HTTP (port 8000, zero overhead, vLLM serves directly)
- KV events use gRPC (port 50051, bridge sidecar translates ZMQ → gRPC)

## Changes (2 files, -11/+4 lines)

```
model_gateway/src/worker/kv_event_monitor.rs        — remove HTTP skip gate
model_gateway/src/workflow/steps/shared/update_policies.rs — remove gRPC-only condition
```

## Build Requirements

### System
- ARM64 (aarch64) host — the K8s cluster runs GB300 ARM64 nodes
- 4+ GB RAM for Rust linking
- ~10 min build time on ARM64

### Dependencies
- Rust toolchain (install via rustup: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`)
- protobuf compiler (`apt install protobuf-compiler` or `brew install protobuf`)
- Build essentials (`apt install build-essential libssl-dev pkg-config`)
- Python 3.12+ with pip
- maturin (`pip install maturin`)

## Build Steps

```bash
# 1. Install Rust if not present
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# 2. Install protoc if not present
# Ubuntu/Debian:
apt install -y protobuf-compiler
# Or download from: https://github.com/protocolbuffers/protobuf/releases

# 3. Install maturin
pip install maturin

# 4. Build the wheel
cd bindings/python
maturin build --release --features vendored-openssl --out dist

# 5. The output wheel will be at:
#    dist/smg-<version>-cp312-cp312-linux_aarch64.whl
ls dist/*.whl
```

## Docker Image Build

After building the wheel, create a Docker image overlay:

```bash
# Create a minimal Dockerfile
cat > /tmp/Dockerfile.smg-dual-port << 'EOF'
FROM 588845226011.dkr.ecr.us-east-2.amazonaws.com/network_ai/mooncake:smg-arm64-5a6dc6995a0e
COPY dist/*.whl /tmp/
RUN pip install --force-reinstall /tmp/smg-*.whl && rm /tmp/smg-*.whl
EOF

# Build
docker build -f /tmp/Dockerfile.smg-dual-port \
  -t 588845226011.dkr.ecr.us-east-2.amazonaws.com/network_ai/mooncake:smg-arm64-dual-port \
  .

# Push
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin 588845226011.dkr.ecr.us-east-2.amazonaws.com
docker push 588845226011.dkr.ecr.us-east-2.amazonaws.com/network_ai/mooncake:smg-arm64-dual-port
```

## Deployment

### SMG Router
- Image: `smg-arm64-dual-port` (the patched image)
- Args:
  ```
  --pd-disaggregation
  --service-discovery
  --service-discovery-namespace=playground
  --service-discovery-port=8000
  --prefill-selector=app=<prefill-label>
  --decode-selector=app=<decode-label>
  --policy=cache_aware
  --tokenizer-path=/tmp/tokenizer
  --host=0.0.0.0
  --port=8080
  ```
- No `--prefill grpc://` workaround needed — uses `http-pd` router naturally
- K8s probes: unchanged (httpGet on port 8080)

### vLLM Prefill/Decode
- Image: unchanged (original vLLM image)
- HTTP on port 8000 as-is
- ZMQ KV events on port 5557 as-is
- Add v1 event-only bridge sidecar:
  - Image: `vllm-mooncake-smg-arm64-c6f5d26` (existing sidecar image)
  - Script: `kv_event_grpc_bridge.py` (mounted via ConfigMap)
  - Port: 50051 (gRPC SubscribeKvEvents only)

### How SMG Connects
- SMG discovers workers on port 8000 (HTTP) via service discovery
- SMG creates HTTP workers → `http-pd` router → HTTP proxy for inference
- KvEventMonitor (with gate removed) tries to subscribe to KV events
- It connects to port 8000 via gRPC → but port 8000 is HTTP → connection fails
- **NOTE**: This patch removes the gate but does NOT add `--kv-event-grpc-port`.
  The KvEventMonitor will attempt gRPC on the worker's HTTP port, which will fail.
  A follow-up change is needed to specify a separate gRPC port for KV events.

### TODO: --kv-event-grpc-port
This patch is step 1 (remove the gate). Step 2 is adding `--kv-event-grpc-port`
to SMG so it connects to port 50051 for gRPC events instead of the worker's
HTTP port. That requires additional Rust changes in:
- `model_gateway/src/main.rs` — add CLI arg
- `model_gateway/src/worker/kv_event_monitor.rs` — use kv_event_grpc_port for gRPC connection
- `model_gateway/src/app_context.rs` — pass the port through config

Without `--kv-event-grpc-port`, the KvEventMonitor will try gRPC on port 8000
(the HTTP worker URL), fail to connect, and retry indefinitely. The event-driven
routing won't work, but the HTTP inference routing will work fine (no regression).

## Verification

After deploying, check SMG logs for:
```
# Should see KV event subscription attempts (even for HTTP workers)
Starting KV event subscription worker_url=http://<pod-ip>:8000

# Without --kv-event-grpc-port, these will fail with transport errors:
Failed to subscribe to KV events, retrying

# HTTP inference should work normally:
started processing request
finished processing request
```
