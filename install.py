import os
import sys
import platform
import torch
import launch
import pkg_resources

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

print("Initializing Riffusion")
riffusion_skip_install = os.environ.get("RIFFUSION_SKIP_INSTALL", False)

if not riffusion_skip_install:
    name = "Riffusion"
    
    # Check if torchaudio is already installed
    try:
        dist = pkg_resources.get_distribution('torchaudio')
        print(f"{name} torchaudio version: {dist.version} is already installed.")
    except pkg_resources.DistributionNotFound:
        print(f"{name} torchaudio is not installed, installing...")
        launch.run(
            f'"{sys.executable}" -m pip install torchaudio',
            f"[{name}] Installing torchaudio...",
            f"[{name}] Couldn't install torchaudio.",
        )

    # Install other requirements
    launch.run(
        f'"{sys.executable}" -m pip install -r "{req_file}"',
        f"[{name}] Installing requirements...",
        f"[{name}] Couldn't install requirements.",
    )