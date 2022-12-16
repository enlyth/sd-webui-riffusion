import os
import sys

from launch import run

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

print("")
print("Initializing Riffusion")
print("")
riffusion_skip_install = os.environ.get("RIFFUSION_SKIP_INSTALL", False)

if not riffusion_skip_install:
    name = "Riffusion"
    run(
        f'"{sys.executable}" -m pip install torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"',
        f"[{name}] Installing torchaudio...",
        f"[{name}] Couldn't torchaudio.",
    )

    run(
        f'"{sys.executable}" -m pip install -r "{req_file}"',
        f"[{name}] Installing requirements...",
        f"[{name}] Couldn't install requirements.",
    )