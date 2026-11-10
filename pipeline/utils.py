"""Utilitaires généraux (bibliothèque standard uniquement)."""
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi")


def in_colab():
    return "google.colab" in sys.modules


def pip_install(*pkgs):
    print("pip install", *pkgs)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    importlib.invalidate_caches()


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def quiet_hf_downloads():
    """Désactive les barres de progression des téléchargements Hugging Face (modèles Whisper, bge-m3, Qwen).

    Dans Jupyter, le téléchargeur lance beaucoup de threads qui mettent à jour une barre `tqdm.notebook` ;
    chaque thread qui écrit ouvre sa propre socket ZeroMQ, jamais refermée. Au-delà de ~1 000 threads,
    le noyau plante avec « zmq.error.ZMQError: Too many open files ». Sans barre, ces threads n'écrivent rien.
    """
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass


def run(cmd, cwd=None):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Commande échouée : {' '.join(map(str, cmd[:4]))} ...\n{r.stderr[-1500:]}")
    return r


def probe(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", path]).stdout
    d = json.loads(out)
    duration = float(d.get("format", {}).get("duration") or 0)
    has_video = any(s.get("codec_type") == "video" for s in d.get("streams", []))
    return duration, has_video


def slugify(s, maxlen=60):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return (s or "x")[:maxlen].strip("-") or "x"


def fmt_time(t):                       # H:MM:SS (affichage console)
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def fmt_clock(t, long_fmt):            # [MM:SS] ou [H:MM:SS] pour les vidéos d'une heure ou plus
    t = int(round(t))
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if long_fmt else f"{m:02d}:{s:02d}"


def to_roman(n):
    out = ""
    for v, r in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                 (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out += r
            n -= v
    return out


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_url(s):
    return bool(re.match(r"^https?://", s.strip()))


def get_secret(name):
    val = os.environ.get(name)
    if not val and in_colab():
        try:
            from google.colab import userdata
            val = userdata.get(name)
        except Exception:
            val = None
    return val
