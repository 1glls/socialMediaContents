"""Thèmes, titres courts et descriptions des sections (LLM ou résumé extractif), fusion des suites, plan texte."""
import json
import math
import re

import numpy as np

from .boundaries import embed_texts, initial_sections
from .llm import LLMClient
from .project import for_each
from .utils import fmt_clock, fmt_time, load_json, save_json, to_roman


# ------------------------- prompt et lecture de la réponse -----------------
def build_prompt(text, ctx_titles, k, n, minutes, cfg):
    start = max(0, len(ctx_titles) - 8)
    prev = "\n".join(f"{j + 1}. {ctx_titles[j]}" for j in range(start, len(ctx_titles))) or "(aucune : c'est le début de l'enregistrement)"
    gloss = f"\nTermes et noms à orthographier ainsi : {cfg.GLOSSARY}\n" if cfg.GLOSSARY.strip() else ""
    p = len(ctx_titles)                      # numéro de la section immédiatement précédente
    suite = (f'- "suite_de" : {p} UNIQUEMENT si cette section poursuit sans rupture le sujet précis de la section {p} '
             f"(la même idée se prolonge) ; sinon null. Un sujet voisin, un même thème général (foi, réveil, Saint-Esprit...) "
             f"ou le retour à une idée évoquée plus tôt ne sont PAS une suite. Dans le doute : null."
             if p else '- "suite_de" : null (première section).')
    return f"""Tu structures la transcription d'un enregistrement (enseignement, prédication ou réunion) en grandes sections thématiques.
Chaque section dure plusieurs minutes : le thème doit exprimer l'idée centrale de TOUTE la section, pas un détail.

Section à analyser : n°{k} sur environ {n} (durée : {minutes} min).{gloss}
Thèmes des sections précédentes :
{prev}

TRANSCRIPTION (reconnaissance vocale automatique, avec d'éventuelles erreurs) :
<<<
{text}
>>>

Rends un objet JSON avec exactement ces clés :
- "theme" : UNE phrase complète et compréhensible (10 à 22 mots), en {cfg.OUTPUT_LANGUAGE}, qui énonce l'idée centrale de la section. Pas de liste de mots-clés, pas de titre télégraphique, pas de numérotation.
- "titre_court" : titre de chapitre YouTube en {cfg.OUTPUT_LANGUAGE}, {cfg.CHAPTER_TITLE_MAX} caractères maximum, concret et propre à CETTE section (ce qui la distingue des autres sections), sans numérotation, sans guillemets, sans point final.
- "description" : résumé en {cfg.OUTPUT_LANGUAGE} de 3 à 5 phrases : idées principales, arguments, exemples et références (textes, personnes, chiffres) effectivement cités. N'invente rien qui ne soit pas dans le texte.
{suite}

Réponds UNIQUEMENT avec l'objet JSON."""


def parse_llm_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    th = re.search(r'"theme"\s*:\s*"((?:[^"\\]|\\.)*)"', txt, re.S)
    ds = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)', txt, re.S)
    if th and ds:
        return {"theme": th.group(1), "description": ds.group(1).rstrip('"} \n'), "suite_de": None}
    return None


def clean_text(s):
    s = re.sub(r"\s+", " ", str(s or "")).strip().strip('"«» ')
    return (s[:1].upper() + s[1:]) if s else s


def section_text(units, sec, cfg):
    text = " ".join(u["text"] for u in units[sec["a"]:sec["b"]])
    if len(text) > cfg.LLM_MAX_CHARS:
        h = int(cfg.LLM_MAX_CHARS * 0.5)
        text = text[:h] + "\n[...]\n" + text[-(cfg.LLM_MAX_CHARS - h):]
    return text


# ------------------------- repli extractif (sans LLM) ------------------------
def extractive_describe(units, sec, cfg, env):
    full = " ".join(u["text"] for u in units[sec["a"]:sec["b"]])
    sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", full) if 6 <= len(s.split()) <= 45]
    if not sents:
        sents = [u["text"] for u in units[sec["a"]:sec["b"]] if u["text"].strip()] or [full[:200]]
    if len(sents) > 300:
        sents = sents[:: math.ceil(len(sents) / 300)]
    E = embed_texts(sents, cfg, env)
    score = E @ E.mean(0)
    theme = sents[int(np.argmax(score))]
    words = theme.split()
    if len(words) > 24:
        theme = " ".join(words[:24]) + "…"
    top = sorted(np.argsort(-score)[:4])
    return {"theme": clean_text(theme), "short_title": "", "description": " ".join(sents[i] for i in top),
            "suite_de": None, "method": "extractif"}


def describe_section(units, sec, ctx_titles, k, n, cfg, env, llm):
    if llm.mode != "none":
        prompt = build_prompt(section_text(units, sec, cfg), ctx_titles, k, n, round((sec["end"] - sec["start"]) / 60), cfg)
        for attempt in range(2):
            try:
                d = parse_llm_json(llm.call(prompt if attempt == 0 else prompt + "\nAttention : réponds uniquement par le JSON demandé."))
            except Exception as e:
                print("  (LLM) échec :", str(e)[:150])
                d = None
            if d and clean_text(d.get("theme")) and clean_text(d.get("description")):
                sd = d.get("suite_de")
                sd = int(sd) if isinstance(sd, (int, float)) or str(sd).strip().isdigit() else None
                return {"theme": clean_text(d["theme"]), "short_title": clean_text(d.get("titre_court")).rstrip(". "),
                        "description": clean_text(d["description"]), "suite_de": sd, "method": "llm"}
        print("  (LLM) réponse inexploitable : repli extractif pour cette section.")
    return extractive_describe(units, sec, cfg, env)


# ------------------------- assemblage des sections ---------------------------
def assemble_sections(units, raw, cfg, env, llm):
    results = []
    for idx, s in enumerate(raw, 1):
        themes = [r["theme"] for r in results]
        d = describe_section(units, s, themes, idx, len(raw), cfg, env, llm)
        prev = results[-1] if results else None
        continues = prev is not None and d["suite_de"] == len(results)      # suite directe de la section précédente uniquement
        if prev is not None and not continues:                               # garde-fou : thèmes quasi identiques
            sim = float(np.dot(*embed_texts([prev["theme"], d["theme"]], cfg, env)))
            if sim >= cfg.THEME_SIM_CONTINUE:
                continues = True
                print(f"  section {idx} : thème quasi identique au précédent (similarité {sim:.2f}) -> suite")
        if cfg.MERGE_CONTINUING and continues and s["end"] - prev["start"] <= cfg.MAX_SECTION_SEC:
            merged = {**prev, "end": s["end"], "b": s["b"]}                # même thème : on fusionne puis on rédige à nouveau
            d2 = describe_section(units, merged, themes[:-1], len(results), len(raw), cfg, env, llm)
            n_prev = len(results) - 1                                      # sections avant la section fusionnée
            merged.update(theme=d2["theme"], short_title=d2["short_title"], description=d2["description"], method=d2["method"],
                          part_of=(n_prev - 1) if n_prev and d2["suite_de"] == n_prev else prev.get("part_of"))
            results[-1] = merged
            print(f"  section {idx}/{len(raw)} fusionnée avec la précédente")
        else:
            results.append({**s, "theme": d["theme"], "short_title": d["short_title"], "description": d["description"],
                            "method": d["method"], "part_of": len(results) - 1 if continues else None})
            print(f"  section {idx}/{len(raw)} : {fmt_time(s['start'])}-{fmt_time(s['end'])}")
    return finalize_titles(results, cfg)


def short_title_of(r, cfg):
    t = clean_text(r.get("short_title")) or r["theme"]
    if len(t) > cfg.CHAPTER_TITLE_MAX:                                     # coupe au dernier mot entier
        t = t[:cfg.CHAPTER_TITLE_MAX - 1].rsplit(" ", 1)[0].rstrip(",;:-–— ") + "…"
    return t


def finalize_titles(sections, cfg):
    """Titres affichés. Chaque section garde SON thème ; « partie k/m » seulement pour une suite de sections
    consécutives sur le même sujet (part_of = section immédiatement précédente)."""
    chains = []
    for i, r in enumerate(sections):
        if chains and r.get("part_of") == i - 1:
            chains[-1].append(i)
        else:
            chains.append([i])
    for chain in chains:
        m = len(chain)
        for rank, i in enumerate(chain, 1):
            r = sections[i]
            r["id"] = i + 1
            r["theme_display"] = r["theme"] + (f" (partie {rank}/{m})" if m > 1 else "")
            r["chapter_title"] = short_title_of(r, cfg) + (f" ({rank}/{m})" if m > 1 else "")
    return sections


def build_plan(sections, long_fmt):
    lines = []
    for r in sections:
        lines += [f"{to_roman(r['id'])}. Thème : {r['theme_display']}",
                  f"Section : [{fmt_clock(r['start'], long_fmt)} - {fmt_clock(r['end'], long_fmt)}]",
                  f"Description : {r['description']}", ""]
    return "\n".join(lines)


# ------------------------- étape 5 : sections d'un projet -------------------------
def build_sections(cfg, env, projects):
    """Étape 5 : frontières, thèmes, descriptions, plan texte ; remplit `project.sections`.

    Le LLM n'est chargé qu'une fois (à la première vidéo qui en a besoin) et libéré à la fin.
    """
    llm = LLMClient(cfg, env)
    loaded = []

    def one(project):
        units = project.units
        sections = []
        if cfg.RUN_THEMES and not (project.sections_json.exists() and "themes" not in cfg.FORCE):
            raw = initial_sections(units, cfg, env)
            print(f"{len(raw)} sections initiales (durées : {[round((s['end'] - s['start']) / 60, 1) for s in raw]} min)")
            if not loaded:
                llm.init()
                loaded.append(True)
            sections = assemble_sections(units, raw, cfg, env, llm)
            save_json(project.sections_json, {"source": project.source.name, "duration": project.duration,
                                              "params": {"min": cfg.MIN_SECTION_SEC, "target": cfg.TARGET_SECTION_SEC,
                                                         "max": cfg.MAX_SECTION_SEC, "embed_model": cfg.EMBED_MODEL,
                                                         "llm": llm.mode},
                                              "sections": [{k: v for k, v in r.items()} for r in sections]})
            project.plan_txt.write_text(build_plan(sections, project.long_fmt), encoding="utf-8")
        elif project.sections_json.exists():
            sections = finalize_titles(load_json(project.sections_json)["sections"], cfg)
            project.plan_txt.write_text(build_plan(sections, project.long_fmt), encoding="utf-8")
            print("Sections déjà présentes : réutilisées (FORCE = {'themes'} pour les refaire).")
        project.sections = sections
        if sections:
            print("\n" + build_plan(sections, project.long_fmt))
            print(f"Plan enregistré : {project.plan_txt}")

    try:
        for_each(projects, "sections et thèmes", one)
    finally:
        if loaded:
            llm.free()
