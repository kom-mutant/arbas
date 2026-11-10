#!/usr/bin/env bash
set -euo pipefail

if [ "$(uname -sm)" != "Linux x86_64" ]; then
  echo 'The training wheels require Linux x86_64 with CUDA 12.9 support.' >&2
  exit 1
fi

cd "$(dirname "$0")/../.."
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT

install_training() {
  # Backport veRL #6551 while retaining the 0.8 trainer API used by Agent Lightning.
  curl -fsSL https://files.pythonhosted.org/packages/3b/f7/d5ff07531e10f0278d3f2f5bf9a0a9ddd5d8d3d3eb903e2dd34c70bee28f/verl-0.8.0.tar.gz \
    -o "$build_dir/verl-0.8.0.tar.gz"
  (cd "$build_dir" && sha256sum -c - <<'SHA256'
814352ab55179dab0724091313e0367e866af667f6a749cc248b73699eb15a21  verl-0.8.0.tar.gz
SHA256
  )
  tar -xzf "$build_dir/verl-0.8.0.tar.gz" -C "$build_dir"
  patch -d "$build_dir/verl-0.8.0" -p1 < "$build_dir/verl.patch"
  uv build --quiet --python 3.12 --wheel --out-dir "$build_dir" "$build_dir/verl-0.8.0"

  uv venv --python 3.12 --allow-existing .venv
  uv --config-file "$build_dir/training-uv.toml" pip sync --python .venv/bin/python --torch-backend cu129 \
    --find-links "$build_dir" "$build_dir/requirements-training.txt"
  uv pip check --python .venv/bin/python
}

# Exact dependency pins, wheel metadata and the veRL patch used above.
write_training_inputs() {
  cat > "$build_dir/requirements-training.txt" <<'REQUIREMENTS'
# Generated with uv pip compile for Linux x86_64, Python 3.12 and CUDA 12.9.
-e .
absl-py==2.5.1
accelerate==1.15.0
agent-detector==2.0.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiohttp-cors==0.8.1
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anthropic==2.7.0
antlr4-python3-runtime==4.9.3
anyio==4.16.0
apache-tvm-ffi==0.1.9
astor==0.8.1
attrs==26.1.0
blake3==1.0.11
cachetools==8.0.0
causal-conv1d @ https://github.com/astral-sh-build/build-causal-conv1d/releases/download/v1.6.2.post1-r1/causal_conv1d-1.6.2.post1%2Bcu.12.9.torch.2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl#sha256=ad3048244ca28fcbb8c797ac51447d5949838d8bfec41da93acb48f44497de99
cbor2==6.2.1
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.1
cloudpickle==3.1.2
codetiming==1.4.0
colorful==0.5.8
compressed-tensors==0.17.0
cryptography==50.0.2
cuda-bindings==12.9.10
cuda-pathfinder==1.8.6
cuda-python==12.9.10
cuda-tile==1.3.0
cuda-toolkit==12.9.1
cwevals==0.0.1b1
datasets==5.2.0
depyf==0.20.0
detect-installer==0.2.1
dill==0.4.1
diskcache==5.6.3
distlib==0.4.4
distro==1.9.0
dnspython==2.9.2
docker==7.2.0
docstring-parser==0.18.0
einops==0.8.2
email-validator==2.3.0
fastapi==0.136.3
fastapi-cli==0.0.32
fastapi-cloud-cli==0.27.1
fastar==0.12.0
fastsafetensors==0.5.0
filelock==4.2.10
fla-core==0.6.0
flash-attn @ https://github.com/astral-sh-build/build-flash-attention/releases/download/v2.8.3-r4/flash_attn-2.8.3%2Bcu.12.9.torch.2.11-cp312-cp312-manylinux_2_24_x86_64.whl#sha256=5a682548f167afc87a5e14118b1d528aaac8681487a11adda3129466a210d79a
flash-linear-attention==0.6.0
flashinfer-cubin==0.6.12
flashinfer-python==0.6.12
frozenlist==1.8.0
fsspec==2026.9.0
google-api-core==2.44.0
google-auth==2.62.1
googleapis-common-protos==1.76.0
grpcio==1.85.1
h11==0.16.0
hf-xet==1.8.0
httpcore==1.0.9
httpcore2==2.16.0
httptools==0.9.0
httpx==0.28.1
httpx-retries==0.6.0
httpx-ws==0.9.0
httpx2==2.16.0
huggingface-hub==1.33.0
humming-kernels==0.1.6
hydra-core==1.3.2
idna==3.20
ijson==3.6.0
importlib-metadata==10.2.0
interegular==0.3.3
jinja2==3.1.6
jiter==0.19.0
jmespath==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kr8s==0.20.15
lark==1.2.2
lazy-importer==0.1.0
llguidance==1.7.6
llvmlite==0.47.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown==3.11.3
markdown-it-py==4.2.0
markupsafe==3.0.4
mcp==2.3.0
mcp-types==2.3.0
mdurl==0.1.2
mistral-common==1.13.1
ml-dtypes==0.6.0
model-hosting-container-standards==0.1.16
mpmath==1.3.0
msgpack==1.2.4
msgspec==0.22.0
multidict==7.2.1
multiprocess==0.70.19
networkx==3.7
ninja==1.13.2
numba==0.65.0
numpy==2.3.5
nvidia-cublas-cu12==12.9.1.4
nvidia-cuda-cccl-cu12==12.9.27
nvidia-cuda-crt==13.5.37
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvcc==13.2.86
nvidia-cuda-nvcc-cu12==12.9.86
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime==13.5.37
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cuda-tileiras==13.2.86
nvidia-cudnn-cu12==9.17.1.4
nvidia-cudnn-frontend==1.28.0
nvidia-cufft-cu12==11.4.1.4
nvidia-cufile-cu12==1.14.1.1
nvidia-curand-cu12==10.3.10.19
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-cusparselt-cu12==0.7.1
nvidia-cutlass-dsl==4.5.2
nvidia-cutlass-dsl-libs-base==4.5.2
nvidia-ml-py==13.615.71
nvidia-nccl-cu12==2.28.9
nvidia-nvjitlink==13.5.37
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.9.79
nvidia-nvvm==13.2.86
omegaconf==2.3.0
openai==2.54.0
openai-harmony==0.0.8
opencensus==0.11.4
opencensus-context==0.1.3
opencv-python-headless==5.0.0.93
opentelemetry-api==1.45.2
opentelemetry-exporter-http-transport==0.66b2
opentelemetry-exporter-otlp==1.45.2
opentelemetry-exporter-otlp-common==0.66b2
opentelemetry-exporter-otlp-proto-common==1.45.2
opentelemetry-exporter-otlp-proto-grpc==1.45.2
opentelemetry-exporter-otlp-proto-http==1.45.2
opentelemetry-exporter-prometheus==0.66b2
opentelemetry-proto==1.45.2
opentelemetry-sdk==1.45.2
opentelemetry-semantic-conventions==0.66b2
opentelemetry-semantic-conventions-ai==0.5.1
orjson==3.13.0
outlines-core==0.2.14
packaging==25.0
pandas==3.1.1
partial-json-parser==0.2.1.1.post7
peft==0.21.2
pillow==13.0.0
platformdirs==4.14.5
prometheus-client==0.26.0
prometheus-fastapi-instrumentator==8.1.0
propcache==0.5.4
proto-plus==1.29.0
protobuf==7.36.2
psutil==7.2.2
py-cpuinfo==9.0.0
py-spy==0.4.3
pyarrow==26.0.1
pyasn1==0.6.5
pyasn1-modules==0.4.2
pybase64==1.5.1
pybind11==3.1.0
pycountry==26.2.16
pycparser==3.12
pydantic==2.14.3
pydantic-core==2.50.4
pydantic-extra-types==2.11.1
pydantic-settings==2.16.2
pyelftools==0.33
pygments==2.21.0
pyjwt==2.15.1
pylatexenc==2.11
python-box==8.0.0
python-dateutil==4.0.0
python-discovery==1.9.5
python-dotenv==1.2.4
python-json-logger==4.2.0
python-jsonpath==2.2.2
python-multipart==0.0.33
pyvers==0.1.0
pyyaml==6.0.3
pyzmq==27.3.0
quack-kernels==0.5.0
ray==2.60.0
referencing==0.37.0
regex==2026.11.9
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.6
rignore==0.8.1
rpds-py==2026.9.1
safetensors==0.8.0
sentencepiece==0.2.3
sentry-sdk==2.74.0
setproctitle==1.3.8
setuptools==80.10.2
shellingham==1.5.4
six==1.17.0
smart-open==9.0.3
sniffio==1.3.1
sse-starlette==3.7.0
starlette==1.10.0
structlog==26.1.0
supervisor==4.3.0
sympy==1.14.0
tabulate==0.10.0
tensorboard==2.21.0
tensorboard-data-server==0.7.2
tensordict==0.10.0
tiktoken==0.14.0
tilelang==0.1.9
tokenizers==0.22.2
tokenspeed-mla==0.1.2
tokenspeed-triton==3.9.10.post20261028
torch==2.11.0+cu129
torch-c-dlpack-ext==0.1.5
torchaudio==2.11.0+cu129
torchdata==0.11.0
torchvision==0.26.0+cu129
tqdm==4.70.1
transformers==5.10.4
triton==3.6.0
truststore==0.10.5
typer==0.29.0
typing-extensions==4.16.0
typing-inspection==0.4.5
tzdata==2026.6
urllib3==2.8.0
uvicorn==0.54.0
uvloop==0.23.0
verl==0.8.0+arbas.2
virtualenv==21.14.5
vllm @ https://wheels.vllm.ai/ee0da84ab9e04ac7610e28580af62c365e898389/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl
wandb==0.31.0
watchfiles==1.3.0
websockets==17.4
werkzeug==3.1.9
wrapt==2.5.2
wsproto==1.3.2
xgrammar==0.2.1
xxhash==4.1.0
yarl==1.25.1
z3-solver==4.15.4.0
zipp==4.1.2
REQUIREMENTS

  cat > "$build_dir/training-uv.toml" <<'UV_CONFIG'
# Metadata from the exact CUDA wheels pinned above.
# This lets uv resolve the environment before downloading the wheel payloads.
[[dependency-metadata]]
name = "flash-attn"
version = "2.8.3+cu.12.9.torch.2.11"
requires-python = ">=3.9"
requires-dist = ["torch ==2.11.*", "einops"]

[[dependency-metadata]]
name = "causal-conv1d"
version = "1.6.2.post1+cu.12.9.torch.2.11"
requires-python = ">=3.9"
requires-dist = ["torch ==2.11.*", "packaging", "ninja"]
UV_CONFIG

  cat > "$build_dir/verl.patch" <<'VERL_PATCH'
--- a/setup.py
+++ b/setup.py
@@ -32 +32 @@
-    "numpy<2.0.0",
+    "numpy>=2.0.0",
@@ -41 +41 @@
-    "transformers",
+    "transformers!=5.6.0",
--- a/verl/protocol.py
+++ b/verl/protocol.py
@@ -217 +217 @@
-        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))
+        non_tensor[key] = np.reshape(val, (new_batch_size, -1, *val.shape[1:]))
@@ -236 +236 @@
-        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))
+        non_tensor_new[key] = np.reshape(val, (batch_size, *val.shape[batch_dims:]))
--- a/verl/version/version
+++ b/verl/version/version
@@ -1 +1 @@
-0.8.0
+0.8.0+arbas.2
--- a/verl/workers/rollout/vllm_rollout/vllm_rollout.py
+++ b/verl/workers/rollout/vllm_rollout/vllm_rollout.py
@@ -103,3 +103,2 @@
-        # multiple replicas share a node, and the Ray job id so two independent
-        # verl jobs on the same host (or a new run after a crashed one with a
-        # stale socket file) cannot collide on the shared /tmp namespace.
+        # multiple replicas share a node. Independent Ray instances can reuse
+        # the same job id, so use the private directory from the launcher.
@@ -108 +107,2 @@
-        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"
+        ipc_dir = os.environ.get("VERL_WEIGHT_TRANSFER_IPC_DIR", "/tmp")
+        self.zmq_handle = f"ipc://{ipc_dir}/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"
--- a/verl/workers/rollout/vllm_rollout/utils.py
+++ b/verl/workers/rollout/vllm_rollout/utils.py
@@ -294,4 +294,2 @@
-        avoids collisions when multiple replicas share the same node, and is
-        unique per Ray job to avoid cross-job collisions on shared hosts. The
-        job id is forwarded by the vLLMHttpServer actor as VERL_RAY_JOB_ID and
-        inherited by this vLLM worker subprocess.
+        and avoids collisions when multiple replicas share the same node. The
+        vLLMHttpServer actor forwards VERL_RAY_JOB_ID to this worker subprocess.
@@ -301 +299,2 @@
-        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"
+        ipc_dir = os.environ.get("VERL_WEIGHT_TRANSFER_IPC_DIR", "/tmp")
+        return f"ipc://{ipc_dir}/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"
VERL_PATCH
}

write_training_inputs
install_training
