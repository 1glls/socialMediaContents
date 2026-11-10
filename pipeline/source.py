"""Sources : fichier, dossier de vidéos, URL YouTube (vidéo ou playlist) ou téléversement Colab.

`resolve_sources` retourne une liste de `Project` (un par vidéo) et prépare l'audio de chacun.
"""
import shutil
from pathlib import Path

from .download import download_url
from .project import Project, for_each
from .utils import VIDEO_EXTS, fmt_time, is_url, probe, run, slugify


def _upload_colab(env):
    from google.colab import files
    up = files.upload()
    if not up:
        raise RuntimeError("Aucun fichier téléversé.")
    name = next(iter(up))
    dest = env.work / "uploads"
    dest.mkdir(exist_ok=True)
    shutil.move(name, dest / name)
    return (dest / name).resolve()


def list_videos(folder):
    """Toutes les vidéos d'un dossier (sous-dossiers compris), triées par chemin."""
    found = [p for p in Path(folder).rglob("*")
             if p.is_file() and p.suffix.lower() in VIDEO_EXTS and "sections_video" not in p.parts]
    return sorted(found)


def make_project(source, env, group=None):
    """Crée le dossier de résultats d'une vidéo : outputs/<titre> ou outputs/<playlist>/<titre>."""
    if source.name == "audio.wav":                  # un audio.wav d'une exécution précédente : son dossier devient le projet
        directory = source.parent
    else:
        directory = env.work / slugify(group) / slugify(source.stem) if group else env.work / slugify(source.stem)
    directory.mkdir(parents=True, exist_ok=True)
    duration, has_video = probe(source)
    return Project(source=source, dir=directory, duration=duration, has_video=has_video, group=group)


def _group_of(source, base):
    """Nom de la playlist / du dossier d'origine, sauf pour une vidéo posée directement dans `base`."""
    parent = source.parent.resolve()
    return None if parent == Path(base).resolve() else parent.name


def prepare_audio(cfg, project):
    """Extrait l'audio mono 16 kHz si la transcription en a besoin."""
    if cfg.RUN_TRANSCRIBE and not cfg.TRANSCRIPT_FILE and not project.transcript_json.exists() or "audio" in cfg.FORCE:
        if project.source != project.audio and (not project.audio.exists() or "audio" in cfg.FORCE):
            print("Extraction audio (mono, 16 kHz)...")
            run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", project.source, "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", project.audio])


def _find_sources(cfg, env):
    """Retourne [(fichier, groupe)] selon la nature de INPUT."""
    src = cfg.INPUT.strip().strip('"').strip("'")
    if not src:
        if env.in_colab:
            return [(_upload_colab(env), None)]
        raise ValueError("Renseigne INPUT (fichier, dossier ou URL YouTube) dans la cellule de configuration.")
    if is_url(src):
        paths, failed = download_url(src, env.download_dir, playlist=cfg.PLAYLIST, items=cfg.PLAYLIST_ITEMS or None)
        if failed:
            print("Certaines vidéos n'ont pas pu être téléchargées (privées, supprimées, bloquées...) : voir les erreurs ci-dessus.")
        if not paths:
            raise RuntimeError("Aucune vidéo téléchargée.")
        return [(p.resolve(), _group_of(p, env.download_dir)) for p in paths]
    p = Path(src).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        videos = list_videos(p)
        if not videos:
            raise FileNotFoundError(f"Aucune vidéo ({', '.join(VIDEO_EXTS)}) dans {p}")
        return [(v.resolve(), _group_of(v, env.download_dir)) for v in videos]
    return [(p.resolve(), None)]


def resolve_sources(cfg, env):
    """Télécharge au besoin, puis retourne un `Project` par vidéo (audio extrait, durée mesurée)."""
    sources = _find_sources(cfg, env)
    if len(sources) > 1 and cfg.TRANSCRIPT_FILE:
        raise ValueError("TRANSCRIPT_FILE ne s'applique qu'à une seule vidéo : laisse-le vide pour une playlist ou un dossier.")
    projects = []
    for source, group in sources:
        try:
            projects.append(make_project(source, env, group))
        except Exception as e:
            if len(sources) == 1:
                raise
            print(f"  Source ignorée ({source.name}) : {str(e)[:200]}")
    for_each(projects, "audio", lambda p: prepare_audio(cfg, p))

    if len(projects) > 1:
        print(f"\n{len(projects)} vidéos à traiter" + (f" (playlist : {projects[0].group})" if projects[0].group else "") + " :")
    for i, p in enumerate(projects, 1):
        if len(projects) > 1:
            print(f"\n[{i}/{len(projects)}]")
        print(f"Source : {p.source.name}")
        print(f"Projet : {p.dir}")
        print(f"Durée  : {fmt_time(p.duration)} | vidéo : {'oui' if p.has_video else 'non (audio seul : découpage vidéo impossible)'}")
    return projects
