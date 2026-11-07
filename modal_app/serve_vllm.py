"""OpenAI-compatible vLLM server for openai/gpt-oss-20b on Modal.

Cost controls (Modal credits are limited):
  * scaledown_window=300  -> the GPU container shuts down after 5 idle minutes
  * max_containers=1      -> never more than one GPU at a time
  * weights + compile caches live on Modal Volumes -> no re-download on cold start
  * scripts stop the app explicitly (`modal app stop rem-vllm`) after each batch

Deploy:   modal deploy modal_app/serve_vllm.py      (needs REM_VLLM_KEY in the local env)
URL:      printed by deploy; put it in .env as REM_VLLM_URL=<url>/v1
"""
import os
import subprocess

import modal

MODEL = "openai/gpt-oss-20b"
GPU = os.environ.get("REM_GPU", "H100")
VLLM_VERSION = "0.11.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "transformers==4.57.1", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)
hf_cache = modal.Volume.from_name("rem-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("rem-vllm-cache", create_if_missing=True)

app = modal.App("rem-vllm")


@app.function(image=image, volumes={"/root/.cache/huggingface": hf_cache}, timeout=60 * 60)
def download():
    """CPU-only: pull weights into the volume so no GPU time is spent downloading."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, ignore_patterns=["*.pth", "original/*", "metal/*"])
    hf_cache.commit()


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=300,
    max_containers=1,
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_dict({"REM_VLLM_KEY": os.environ.get("REM_VLLM_KEY", "")})],
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["REM_VLLM_KEY"],
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "256",
        "--enable-prefix-caching",
        "--tool-call-parser", "openai", "--enable-auto-tool-choice",
    ]
    proc = subprocess.Popen(cmd)

    def watchdog():
        # fail fast: if vLLM dies (e.g. bad dependency), kill the container so no idle GPU is billed
        code = proc.wait()
        print(f"vLLM exited with code {code}; terminating container", flush=True)
        os._exit(1)

    import threading
    threading.Thread(target=watchdog, daemon=True).start()
