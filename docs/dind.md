# Docker-in-Docker vs Docker Socket

## Problem
Running `docker run` inside a container fails: **no Docker daemon in container**.

## Solutions

### ✅ Recommended: Mount Docker Socket
```yaml
services:
  api-server:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # KEY!
```

**Pros:**
- Simple, no privileged mode needed
- Good performance, no extra overhead
- Training containers run as siblings on host

**Cons:**
- Security risk: container can control all host containers
- Requires careful permission management

**How it works:**
- Container uses host's Docker daemon
- Training containers run as **siblings**, not nested
- No special privileges needed

**Python Example:**
```python
import docker
client = docker.from_env()

container = client.containers.run(
    image="llamafactory:latest",
    command=["llamafactory-cli", "train", "config.yaml"],
    device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
    detach=True
)
```

### ⚠️ Not Recommended: Docker-in-Docker (DinD)
```yaml
services:
  api-server:
    image: docker:dind
    privileged: true  # Required!
```

**Why avoid:**
- Requires `--privileged` mode (security risk)
- Complex GPU passthrough
- Performance overhead

## Architecture

```
Host
├── API Server Container (mounts docker.sock)
├── Training Container 1  ← Created by API Server
└── Training Container 2  ← Created by API Server
```

## Security Note
Mounting docker.sock gives container **full control** over host Docker. Use with caution in production.
