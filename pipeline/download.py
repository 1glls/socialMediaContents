"""Téléchargement YouTube (vidéo ou playlist) avec yt-dlp, fichiers nommés d'après leur titre.

Nommage :
    vidéo seule : <dossier>/<titre>.mp4
    playlist    : <dossier>/<titre de la playlist>/<NN> - <titre>.mp4
"""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


def import_yt_dlp():
    """Importe yt_dlp, en l'installant au besoin."""
    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp introuvable : installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"])
        importlib.invalidate_caches()
        import yt_dlp
    return yt_dlp


def build_options(out_dir, *, playlist, audio_only, max_height, items, is_playlist, paths):
    """Construit les options yt-dlp (nommage, format, robustesse)."""
    if is_playlist:
        # Un sous-dossier par playlist ; l'index (complété par des zéros) garde l'ordre de la liste.
        # ".150B" limite la longueur du titre seul (en octets) pour éviter les noms de fichiers trop longs.
        template = "%(playlist_title|Playlist).100B/%(playlist_index)s - %(title).150B.%(ext)s"
    else:
        template = "%(title).150B.%(ext)s"

    opts = {
        "outtmpl": str(out_dir / template),
        "noplaylist": not playlist,        # watch?v=X&list=Y : la vidéo seule, sauf si playlist=True
        "windowsfilenames": True,          # retire les caractères interdits (: ? " / ...) des noms
        "noprogress": True,                # pas de barre de progression (illisible dans un log)
        "ignoreerrors": True,              # une vidéo privée/supprimée n'arrête pas la playlist
        "retries": 5,
        "post_hooks": [lambda path: paths.append(Path(path))],   # chemin final de chaque fichier
    }
    if items:
        opts["playlist_items"] = items
    if audio_only:
        opts["format"] = "bestaudio/best"
        opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}]
    else:
        opts["format"] = f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"
        opts["merge_output_format"] = "mp4"
    return opts


def download_url(url, out_dir, *, playlist=False, audio_only=False, max_height=1080, items=None):
    """Télécharge une vidéo ou une playlist. Retourne (liste des fichiers, nombre d'échecs)."""
    yt_dlp = import_yt_dlp()
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg introuvable (nécessaire pour fusionner vidéo et audio). Linux : sudo apt install ffmpeg")
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Repérer s'il s'agit d'une playlist (sans rien télécharger).
    probe_opts = {"noplaylist": not playlist, "quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        # Pas de process=False : une URL watch?v=..&list=.. renvoie alors une simple redirection ("url").
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError(f"Impossible de lire l'URL : {url}")
    is_playlist = info.get("_type") in ("playlist", "multi_video")
    if is_playlist:
        title = info.get("title") or "Playlist"
        print(f"Playlist : {title}")
    else:
        print(f"Vidéo : {info.get('title', url)}")

    # 2. Télécharger.
    paths = []
    opts = build_options(out_dir, playlist=playlist, audio_only=audio_only, max_height=max_height,
                         items=items, is_playlist=is_playlist, paths=paths)
    with yt_dlp.YoutubeDL(opts) as ydl:
        failed = ydl.download([url])       # code retour : 0 = tout est passé, 1 = au moins une erreur
    return paths, failed
