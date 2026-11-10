"""Découpe de la vidéo : un fichier par section, chapitres YouTube, copie avec chapitres intégrés."""
import re

from .project import for_each
from .utils import load_json, run, save_json, slugify

# Conteneurs qu'on peut copier tels quels (sans ré-encodage) ; sinon on sort en .mkv
KEEP_EXTS = (".mp4", ".mkv", ".mov", ".m4v", ".webm")


def video_bounds(sections, duration):
    out = []
    for i, s in enumerate(sections):
        a = 0.0 if i == 0 else float(s["start"])
        b = float(duration) if i == len(sections) - 1 else float(sections[i + 1]["start"])
        out.append((a, b))
    return out


def yt_ts(t):
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def ffmeta_escape(s):
    return re.sub(r"([=;#\\\n])", lambda m: "\\" + m.group(1), s.replace("\n", " "))


def cut_project(cfg, project):
    from tqdm.auto import tqdm
    if not project.has_video:
        print("Pas de piste vidéo : découpage ignoré.")
        return
    if not project.sections:
        print("Aucune section : lance l'étape 2 (RUN_THEMES = True).")
        return
    source, sections, directory = project.source, project.sections, project.dir
    video_dir = directory / "sections_video"
    bounds = video_bounds(sections, project.duration)
    keep_ext = source.suffix.lower() in KEEP_EXTS
    out_ext = ".mp4" if cfg.CUT_MODE == "reencode" else (source.suffix.lower() if keep_ext else ".mkv")

    # --- chapitres (texte YouTube + métadonnées ffmpeg) ---------------------------------
    (directory / "chapitres_youtube.txt").write_text(
        "\n".join(f"{yt_ts(a)} {r['chapter_title']}" for (a, _), r in zip(bounds, sections)) + "\n", encoding="utf-8")
    meta = [";FFMETADATA1"]
    for (a, b), r in zip(bounds, sections):
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(round(a * 1000))}", f"END={int(round(b * 1000))}",
                 f"title={ffmeta_escape(r['chapter_title'])}"]
    meta_path = directory / "chapitres.ffmetadata"
    meta_txt = "\n".join(meta) + "\n"
    chapters_changed = not meta_path.exists() or meta_path.read_text(encoding="utf-8") != meta_txt
    meta_path.write_text(meta_txt, encoding="utf-8")

    # --- un fichier par section ------------------------------------------------------------
    if cfg.CUT_SECTIONS:
        video_dir.mkdir(exist_ok=True)
        manifest_path = video_dir / "manifest.json"          # bornes de chaque fichier : recoupe si le plan change
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        names = [f"{r['id']:02d}_{slugify(r['chapter_title'], 50)}{out_ext}" for r in sections]
        stale = [p for p in video_dir.iterdir() if p.is_file() and re.match(r"\d{2}_", p.name) and p.name not in names]
        for p in stale:
            p.unlink()
            manifest.pop(p.name, None)
        if stale:
            print(f"{len(stale)} fichier(s) d'un plan précédent supprimé(s).")
        for (a, b), r, name in tqdm(list(zip(bounds, sections, names)), desc="Sections vidéo"):
            out = video_dir / name
            key = [round(a, 3), round(b, 3), cfg.CUT_MODE]
            if out.exists() and manifest.get(name) == key and "video" not in cfg.FORCE:
                continue
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{a:.3f}", "-i", source, "-t", f"{b - a:.3f}",
                   "-map", "0:v:0", "-map", "0:a:0?"]
            if cfg.CUT_MODE == "copy":
                cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
            else:
                cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]
            run(cmd + [out.name], cwd=video_dir)
            manifest[name] = key
            save_json(manifest_path, manifest)

    # --- vidéo complète avec chapitres intégrés -------------------------------------------
    if cfg.EMBED_CHAPTERS:
        chap_ext = source.suffix.lower() if keep_ext else ".mkv"
        out = directory / f"{slugify(source.stem)}_chapitres{chap_ext}"
        if not out.exists() or chapters_changed or "video" in cfg.FORCE:
            run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", source, "-i", directory / "chapitres.ffmetadata",
                 "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "1", "-map_chapters", "1", "-c", "copy", out])
        print("Vidéo avec chapitres :", out.name)

    print("\nChapitres YouTube :")
    print((directory / "chapitres_youtube.txt").read_text(encoding="utf-8"))
    if cfg.CUT_SECTIONS:
        print("Fichiers :", *sorted(p.name for p in video_dir.iterdir()), sep="\n  ")


def cut_video(cfg, env, projects):
    """Étape 7 : chapitres YouTube, un fichier vidéo par section, copie avec chapitres intégrés (par vidéo)."""
    if not cfg.RUN_VIDEO:
        return
    for_each(projects, "découpage vidéo", lambda p: cut_project(cfg, p))
