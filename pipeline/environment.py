"""Préparation de l'environnement : paquets, ffmpeg, GPU/CPU, dossier de travail."""
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import has_module, in_colab, pip_install


@dataclass
class Env:
    in_colab: bool
    has_cuda: bool
    device: str                  # "cuda" ou "cpu"
    gpu_name: str | None
    cpu_threads: int
    whisper_model: str           # nom du modèle Whisper effectivement utilisé
    cpu_fallback_model: str
    work: Path                   # dossier des résultats (outputs/, ou Google Drive sur Colab)
    download_dir: Path           # dossier des vidéos téléchargées (data/)
    torch: Any = None            # module torch, ou None s'il n'est pas installé


def detect_torch():
    try:
        import torch
        return torch, bool(torch.cuda.is_available())
    except ImportError:
        return None, False


def setup_environment(cfg):
    """Installe les paquets manquants, vérifie ffmpeg, détecte le GPU et crée le dossier de travail."""
    colab = in_colab()
    torch, has_cuda = detect_torch()

    # --- paquets nécessaires selon les étapes et le LLM choisis ------------------
    need = {"numpy": "numpy", "soundfile": "soundfile", "tqdm": "tqdm", "requests": "requests"}
    if cfg.RUN_TRANSCRIBE and not cfg.TRANSCRIPT_FILE:
        need["faster_whisper"] = "faster-whisper"
    if cfg.RUN_THEMES:
        need["sentence_transformers"] = "sentence-transformers"
        if cfg.LLM_BACKEND == "transformers" or (cfg.LLM_BACKEND == "auto" and has_cuda):
            need["transformers"] = "transformers"
            need["accelerate"] = "accelerate"
            if has_cuda and cfg.LLM_4BIT:
                need["bitsandbytes"] = "bitsandbytes"
    missing = [pkg for mod, pkg in need.items() if not has_module(mod)]
    if missing and cfg.INSTALL_DEPS:
        pip_install(*missing)
        print("Si Colab/Jupyter signale une incompatibilité de versions, redémarre le noyau puis relance depuis le début.")
    elif missing:
        print("Paquets manquants :", missing)
    if torch is None:
        torch, has_cuda = detect_torch()

    import numpy  # noqa: F401  (échec immédiat et clair si un paquet manque)
    import requests  # noqa: F401
    from tqdm.auto import tqdm  # noqa: F401

    # --- ffmpeg ---------------------------------------------------------------
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("ffmpeg/ffprobe introuvables. Windows : winget install Gyan.FFmpeg | macOS : brew install ffmpeg | "
                           "Linux : sudo apt install ffmpeg (puis redémarre le noyau).")

    device = "cuda" if has_cuda else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else None
    whisper_model = ("large-v3-turbo" if has_cuda else "medium") if cfg.WHISPER_MODEL == "auto" else cfg.WHISPER_MODEL

    # --- dossier de travail -----------------------------------------------------
    if colab and cfg.MOUNT_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        work = Path("/content/drive/MyDrive") / cfg.DRIVE_SUBDIR
    elif colab:
        work = Path("/content/work")
    else:
        work = Path(cfg.WORK_DIR)
    work.mkdir(parents=True, exist_ok=True)
    work = work.resolve()

    # --- dossier des téléchargements (data/ en local ; sur Colab, à côté des résultats pour être conservé) --
    download_dir = work / cfg.DOWNLOAD_DIR if colab else Path(cfg.DOWNLOAD_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)
    download_dir = download_dir.resolve()

    env = Env(in_colab=colab, has_cuda=has_cuda, device=device, gpu_name=gpu_name, cpu_threads=os.cpu_count() or 4,
              whisper_model=whisper_model, cpu_fallback_model="small", work=work, download_dir=download_dir, torch=torch)

    print(f"Plateforme : {'Colab' if colab else 'local'} | calcul : {device}" + (f" ({gpu_name})" if gpu_name else ""))
    print(f"Dossier de travail : {work}")
    print(f"Téléchargements : {download_dir}")
    if not has_cuda:
        print("Pas de GPU CUDA" + (" : active un GPU T4 (Exécution > Modifier le type d'exécution)." if colab else " : Whisper tournera sur CPU (plus lent)."))
    return env


def free_gpu(env):
    """Libère la mémoire GPU (à appeler après avoir supprimé les références à un modèle)."""
    import gc
    gc.collect()
    if env.has_cuda and env.torch is not None:
        env.torch.cuda.empty_cache()
