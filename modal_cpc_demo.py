import modal 

APP_NAME = "character-prefix-conditioning"
VLLM_PORT = 8000
N_GPU = 1
MINUTES = 60

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12.2")
    .apt_install("git")
    .uv_pip_install(
        "torch",
        "huggingface_hub[cli]",
        "vllm==0.9.2",
        "transformers==4.52.4",
        "fastapi==0.115.0",
        "uvicorn[standard]==0.30.6",
        "requests>=2.31.0",
    )
    .env({"VLLM_USE_V1": "0"})
    .add_local_dir(".", remote_path="/root/app", ignore=[".git", ".venv", "__pycache__", "**/*.pyc", "vllm"])
)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.concurrent(
    max_inputs=32
)
@modal.asgi_app()
def cpc_serve():

    import sys
    sys.path.insert(0, "/root/app")
    from run_cpc import create_api_app
    return create_api_app()




