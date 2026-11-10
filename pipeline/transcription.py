"""Transcription : faster-whisper (avec repli CUDA/CPU) ou import d'un sous-titre .vtt / .srt."""
import gc
import glob
import os
import re
import sys
from pathlib import Path

from .environment import free_gpu
from .project import for_each
from .utils import fmt_time, load_json, pip_install, quiet_hf_downloads, save_json


# ------------------------- import .vtt / .srt ------------------------------
def parse_vtt(path):
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    ts = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")
    sec = lambda h, m, s, ms: int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    units = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().split("\n")
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        m = ts.search(lines[idx]) if idx is not None else None
        if not m:
            continue
        body = " ".join(l.strip() for l in lines[idx + 1:] if l.strip())
        stripped = re.sub(r"^<v\s+[^>]+>", "", body)                # Teams : <v Nom>texte</v>
        if stripped == body:
            stripped = re.sub(r"^[^:<>]{1,60}:\s+", "", body)       # Zoom : Nom: texte
        body = re.sub(r"<[^>]+>", "", stripped).strip()
        if body:
            units.append({"start": sec(*m.group(1, 2, 3, 4)), "end": sec(*m.group(5, 6, 7, 8)), "text": body})
    return units


def merge_cues(units, max_gap=1.5, max_len=20.0):
    out = []
    for u in units:
        p = out[-1] if out else None
        if p and u["start"] - p["end"] <= max_gap and u["end"] - p["start"] <= max_len:
            p["end"] = u["end"]
            p["text"] += " " + u["text"]
        else:
            out.append(dict(u))
    return out


# ------------------------- Whisper ---------------------------------------
def preload_cuda_libs():
    if not sys.platform.startswith("linux"):
        return
    import ctypes, site
    roots = set()
    try:
        roots |= {os.path.join(sp, "nvidia") for sp in site.getsitepackages() + [site.getusersitepackages()]}
    except Exception:
        pass
    for pattern in ("libcublasLt.so*", "libcublas.so*", "libcudnn.so*"):
        for root in roots:
            for path in sorted(glob.glob(os.path.join(root, "*", "lib", pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _whisper_pass(audio, duration, model_name, device, compute, cfg, env):
    from faster_whisper import WhisperModel
    from tqdm.auto import tqdm
    quiet_hf_downloads()          # téléchargement du modèle : pas de barre (voir utils.quiet_hf_downloads)
    print(f"Whisper '{model_name}' ({device}, {compute})...")
    try:    # cache local d'abord : évite l'appel (anonyme) au HF Hub et son avertissement HF_TOKEN
        model = WhisperModel(model_name, device=device, compute_type=compute, cpu_threads=env.cpu_threads,
                             local_files_only=True)
    except Exception:   # modèle pas encore téléchargé -> téléchargement normal
        model = WhisperModel(model_name, device=device, compute_type=compute, cpu_threads=env.cpu_threads)
    kwargs = dict(language=cfg.LANGUAGE, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
                  beam_size=5, condition_on_previous_text=False, initial_prompt=cfg.INITIAL_PROMPT or None,
                  word_timestamps=cfg.WORD_TIMESTAMPS)
    if cfg.MULTILINGUAL:
        kwargs["multilingual"] = True
    segs, done = [], 0
    try:
        try:
            seg_iter, info = model.transcribe(audio, **kwargs)
        except TypeError:                          # faster-whisper sans l'option multilingual
            kwargs.pop("multilingual", None)
            seg_iter, info = model.transcribe(audio, **kwargs)
        with tqdm(total=max(1, int(duration)), unit="s", desc="Transcription") as bar:
            for s in seg_iter:
                seg = {"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": s.text.strip()}
                if cfg.WORD_TIMESTAMPS and s.words:
                    seg["words"] = [{"w": w.word.strip(), "s": round(float(w.start), 2), "e": round(float(w.end), 2)}
                                    for w in s.words if w.word.strip()]
                segs.append(seg)
                step = int(min(s.end, duration)) - done
                if step > 0:
                    bar.update(step)
                    done += step
            bar.update(bar.total - done)           # fin de parole/silence final ignoré par le VAD : on complète la barre
        lang = info.language
    finally:
        del model
        gc.collect()
        free_gpu(env)
    return segs, lang, model_name


def run_whisper(audio_path, duration, cfg, env):
    import soundfile as sf
    wav, sr = sf.read(str(audio_path), dtype="float32")      # tableau numpy : pas de décodage PyAV
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    device, compute, name = env.device, ("float16" if env.device == "cuda" else "int8"), env.whisper_model
    if device == "cuda":
        preload_cuda_libs()
    for attempt in range(3):
        try:
            return _whisper_pass(wav, duration, name, device, compute, cfg, env)
        except Exception as e:
            msg = str(e).lower()
            if not (device == "cuda" and any(k in msg for k in ("cublas", "cudnn", "cuda", "libcu"))):
                raise
            print("Problème GPU/CUDA :", str(e)[:200])
            if attempt == 0 and cfg.INSTALL_DEPS:
                pip_install("nvidia-cublas-cu12", "nvidia-cudnn-cu12")
                preload_cuda_libs()
                continue
            print(f"Repli sur le CPU avec le modèle '{env.cpu_fallback_model}'.")
            device, compute, name = "cpu", "int8", env.cpu_fallback_model
    raise RuntimeError("Transcription impossible.")


HALLU = re.compile(r"(sous-titr|amara\.org|merci d.avoir regard|abonnez-vous|thanks for watching|subtitles by)", re.I)


def clean_segments(segs):
    out = []
    for s in segs:
        t = s["text"].strip()
        if not t or (HALLU.search(t) and len(t) < 80):
            continue
        if len(out) >= 2 and t == out[-1]["text"] == out[-2]["text"]:      # boucle de répétition
            continue
        item = {"start": s["start"], "end": s["end"], "text": t}
        if s.get("words"):
            item["words"] = s["words"]
        out.append(item)
    return out


# ------------------------- étape 4 : transcription d'un projet ------------------
def transcribe_project(cfg, env, project):
    """Priorité : transcript.json en cache -> import TRANSCRIPT_FILE (.vtt/.srt) -> faster-whisper."""
    if cfg.RUN_TRANSCRIBE:
        if project.transcript_json.exists() and "transcribe" not in cfg.FORCE:
            print("Transcription déjà présente : réutilisée (FORCE = {'transcribe'} pour la refaire).")
        elif cfg.TRANSCRIPT_FILE:
            units = merge_cues(parse_vtt(cfg.TRANSCRIPT_FILE))
            save_json(project.transcript_json, {"meta": {"origin": "vtt", "duration": project.duration, "language": cfg.LANGUAGE},
                                                "units": units})
        else:
            segs, lang, used_model = run_whisper(project.audio, project.duration, cfg, env)
            units = clean_segments(segs)
            save_json(project.transcript_json, {"meta": {"origin": "whisper", "whisper_model": used_model,
                                                         "duration": project.duration, "language": lang,
                                                         "word_timestamps": any("words" in u for u in units)}, "units": units})

    if not project.transcript_json.exists():
        raise FileNotFoundError(f"{project.transcript_json} introuvable : lance l'étape 1 (RUN_TRANSCRIBE = True) ou copie le fichier ici.")
    project.transcript = load_json(project.transcript_json)
    project.units = project.transcript["units"]
    units, meta = project.units, project.transcript["meta"]
    if cfg.WORD_TIMESTAMPS and meta.get("origin") == "whisper" and not any("words" in u for u in units):
        print("Transcription en cache sans horodatage par mot (modèle %s) : FORCE = {'transcribe'} pour la refaire "
              "avec le modèle et le glossaire actuels." % meta.get("whisper_model"))
    nwords = sum(len(u["text"].split()) for u in units)
    print(f"{len(units)} segments, {nwords} mots | origine : {meta.get('origin')} | langue : {meta.get('language')}")
    for u in units[:5]:
        print(f"[{fmt_time(u['start'])}] {u['text'][:120]}")


def transcribe(cfg, env, projects):
    """Étape 4 : transcrit chaque vidéo (ou réutilise le cache) ; remplit `project.units`."""
    for_each(projects, "transcription", lambda p: transcribe_project(cfg, env, p))
