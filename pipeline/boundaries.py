"""Frontières des sections : blocs de texte, vecteurs, creux de similarité, calage sur les pauses."""
import bisect

import numpy as np


def make_blocks(units, block_sec):
    blocks, cur, t0 = [], None, units[0]["start"]
    for i, u in enumerate(units):
        k = int((u["start"] - t0) // block_sec)
        if cur is None or k != cur["k"]:
            cur = {"k": k, "a": i, "b": i + 1, "start": u["start"], "text": u["text"]}
            blocks.append(cur)
        else:
            cur["b"] = i + 1
            cur["text"] += " " + u["text"]
    return blocks


def similarity_curve(emb, weights, window):
    """s_i = cosinus entre la somme pondérée des `window` blocs avant i et des `window` blocs à partir de i."""
    n = len(emb)
    sims = np.ones(n)
    for i in range(1, n):
        a, b = max(0, i - window), min(n, i + window)
        L = (emb[a:i] * weights[a:i, None]).sum(0)
        R = (emb[i:b] * weights[i:b, None]).sum(0)
        sims[i] = float(L @ R / (np.linalg.norm(L) * np.linalg.norm(R) + 1e-9))
    if n > 1:
        sims[0] = sims[1]
    return sims


def smooth_curve(sims):
    """Moyenne mobile sur 3 points (bords répétés)."""
    return np.convolve(np.pad(sims, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")


def depth_scores(emb, weights, window):
    """Profondeur de creux : d_i = (pic gauche - s_i) + (pic droit - s_i) sur la courbe lissée."""
    n = len(emb)
    sm = smooth_curve(similarity_curve(emb, weights, window))
    depth = np.zeros(n)
    for i in range(1, n):
        l = i
        while l > 0 and sm[l - 1] >= sm[l]:
            l -= 1
        r = i
        while r < n - 1 and sm[r + 1] >= sm[r]:
            r += 1
        depth[i] = (sm[l] - sm[i]) + (sm[r] - sm[i])
    return depth


def pick_boundaries(depth, btimes, t_begin, t_end, min_gap, target, max_len, strictness):
    n = len(depth)
    def admissible(t, chosen):
        return t - t_begin >= min_gap and t_end - t >= min_gap and all(abs(t - btimes[j]) >= min_gap for j in chosen)
    if n < 3:
        return []
    peaks = [i for i in range(1, n) if depth[i] > 0 and depth[i] >= depth[i - 1] and (i == n - 1 or depth[i] >= depth[i + 1])]
    floor = depth[1:].mean() + strictness * depth[1:].std()
    n_target = max(0, int(round((t_end - t_begin) / target)) - 1)
    chosen = []
    for i in sorted(peaks, key=lambda i: -depth[i]):
        if len(chosen) >= n_target or depth[i] < floor:
            break
        if admissible(btimes[i], chosen):
            chosen.append(i)
    while True:                                        # coupures forcées des sections trop longues
        times = sorted([t_begin] + [btimes[j] for j in chosen] + [t_end])
        too_long = [(a, b) for a, b in zip(times[:-1], times[1:]) if b - a > max_len]
        if not too_long:
            break
        a, b = max(too_long, key=lambda ab: ab[1] - ab[0])
        inside = [i for i in range(1, n) if a + min_gap <= btimes[i] <= b - min_gap]
        if inside:
            i = max(inside, key=lambda i: depth[i])
        else:
            mid = (a + b) / 2
            i = min(range(1, n), key=lambda i: abs(btimes[i] - mid))
        if i in chosen:
            break
        chosen.append(i)
    return sorted(chosen)


def snap_to_pause(units, starts, t, radius):
    lo = bisect.bisect_left(starts, t - radius)
    best, best_score = None, -1e9
    for j in range(max(1, lo), len(units)):
        dt = starts[j] - t
        if dt > radius:
            break
        gap = max(0.0, units[j]["start"] - units[j - 1]["end"])
        bonus = 0.5 if units[j - 1]["text"].rstrip().endswith((".", "?", "!", "…")) else 0.0
        score = gap + bonus - 0.01 * abs(dt)
        if score > best_score:
            best, best_score = j, score
    return best


def section_dict(units, a, b):
    return {"a": a, "b": b, "start": units[a]["start"], "end": units[b - 1]["end"]}


# ------------------------- vecteurs de texte --------------------------------
_EMBEDDERS = {}     # (modèle, device) -> SentenceTransformer : chargé une seule fois pour toutes les vidéos


def get_embedder(cfg, env):
    key = (cfg.EMBED_MODEL, env.device)
    if key not in _EMBEDDERS:
        from sentence_transformers import SentenceTransformer
        from .utils import quiet_hf_downloads
        quiet_hf_downloads()
        print(f"Chargement du modèle de vecteurs {cfg.EMBED_MODEL} (premier lancement : téléchargement de plusieurs Go, sans barre de progression)...")
        model = SentenceTransformer(cfg.EMBED_MODEL, device=env.device)
        try:
            model.max_seq_length = min(512, model.max_seq_length or 512)
        except Exception:
            pass
        _EMBEDDERS[key] = model
    return _EMBEDDERS[key]


def embed_texts(texts, cfg, env, show=False):
    return np.asarray(get_embedder(cfg, env).encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=show))


def initial_sections(units, cfg, env):
    t_begin, t_end = units[0]["start"], units[-1]["end"]
    blocks = make_blocks(units, cfg.BLOCK_SEC)
    if len(blocks) < 4 or t_end - t_begin < 2 * cfg.MIN_SECTION_SEC:
        return [section_dict(units, 0, len(units))]
    print(f"{len(blocks)} blocs de {cfg.BLOCK_SEC} s : calcul des vecteurs...")
    emb = embed_texts([b["text"] for b in blocks], cfg, env, show=True)
    weights = np.array([max(1, len(b["text"].split())) for b in blocks], dtype=float)
    btimes = np.array([b["start"] for b in blocks])
    depth = depth_scores(emb, weights, max(2, int(round(cfg.TOPIC_WINDOW_SEC / cfg.BLOCK_SEC))))
    picked = pick_boundaries(depth, btimes, t_begin, t_end, cfg.MIN_SECTION_SEC, cfg.TARGET_SECTION_SEC,
                             cfg.MAX_SECTION_SEC, cfg.TOPIC_STRICTNESS)
    starts = [u["start"] for u in units]
    cuts = []
    for i in picked:
        j = snap_to_pause(units, starts, blocks[i]["start"], cfg.BLOCK_SEC)
        cuts.append(j if j is not None else blocks[i]["a"])
    cuts = sorted(set(c for c in cuts if 0 < c < len(units)))
    bounds = [0] + cuts + [len(units)]
    return [section_dict(units, a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
