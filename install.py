import os
import sys
import platform
import torch
import launch

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

print("Initializing Riffusion")
riffusion_skip_install = os.environ.get("RIFFUSION_SKIP_INSTALL", False)

if not riffusion_skip_install and not launch.is_installed("torchaudio"):
    name = "Riffusion"
    if platform.system() == "Darwin":
        # MacOS
        launch.run(
            f'"{sys.executable}" -m pip install torchaudio==0.13.1',
            f"[{name}] Installing torchaudio...",
            f"[{name}] Couldn't install torchaudio.",
        )
    else:
        if torch.version.hip:
            launch.run(
                f'"{sys.executable}" -m pip install torchaudio==0.13.1+rocm5.2 --extra-index-url https://download.pytorch.org/whl/rocm5.2',
                f"[{name}] Installing torchaudio...",
                f"[{name}] Couldn't install torchaudio.",
            )
        else:
            launch.run(
                f'"{sys.executable}" -m pip install torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117',
                f"[{name}] Installing torchaudio...",
                f"[{name}] Couldn't install torchaudio.",
            )

    launch.run(
        f'"{sys.executable}" -m pip install -r "{req_file}"',
        f"[{name}] Installing requirements...",
        f"[{name}] Couldn't install requirements.",
    )
