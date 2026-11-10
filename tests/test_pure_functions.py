"""Tests des fonctions pures du pipeline (aucun modèle, aucun réseau, aucun ffmpeg)."""
from pathlib import Path

import numpy as np
import pytest

from pipeline import boundaries, themes, transcription, utils, video
from pipeline.config import Config
from pipeline.project import Project, for_each
from pipeline.source import _group_of, list_videos


# ------------------------- utils -------------------------
def test_fmt_time_and_clock():
    assert utils.fmt_time(0) == "0:00:00"
    assert utils.fmt_time(3725.4) == "1:02:05"
    assert utils.fmt_clock(65, False) == "01:05"
    assert utils.fmt_clock(3725, True) == "1:02:05"


def test_to_roman():
    assert [utils.to_roman(n) for n in (1, 4, 9, 14, 40, 90, 400, 1994)] == ["I", "IV", "IX", "XIV", "XL", "XC", "CD", "MCMXCIV"]


def test_slugify():
    assert utils.slugify("Création d'une société (Bob 50)") == "creation-d-une-societe-bob-50"
    assert utils.slugify("") == "x"
    assert utils.slugify("a" * 100, maxlen=10) == "a" * 10


def test_is_url():
    assert utils.is_url(" https://youtu.be/x ")
    assert not utils.is_url("data/video.mp4")


def test_json_roundtrip(tmp_path):
    utils.save_json(tmp_path / "a.json", {"é": [1, 2]})
    assert utils.load_json(tmp_path / "a.json") == {"é": [1, 2]}


# ------------------------- config -------------------------
def test_config_defaults_and_typo():
    cfg = Config()
    assert cfg.TARGET_SECTION_SEC == 480 and cfg.DOWNLOAD_DIR == "data" and cfg.PLAYLIST is False
    with pytest.raises(TypeError):
        Config(TARGET_SECTION_SECS=1)          # faute de frappe : refusée
    assert Config(FORCE=["video"]).FORCE == {"video"}


# ------------------------- transcription -------------------------
VTT = """WEBVTT

1
00:00:01.000 --> 00:00:03.000
<v Marie>Bonjour à tous</v>

2
00:00:03.500 --> 00:00:06.000
Jean: on commence maintenant

3
00:01:00.000 --> 00:01:02.000
<i>Une autre phrase</i>
"""


def test_parse_vtt_and_merge(tmp_path):
    p = tmp_path / "t.vtt"
    p.write_text(VTT, encoding="utf-8")
    units = transcription.parse_vtt(p)
    assert [u["text"] for u in units] == ["Bonjour à tous", "on commence maintenant", "Une autre phrase"]
    assert units[2]["start"] == 60.0
    merged = transcription.merge_cues(units)
    assert len(merged) == 2 and merged[0]["text"] == "Bonjour à tous on commence maintenant" and merged[0]["end"] == 6.0


def test_clean_segments():
    segs = [{"start": 0, "end": 1, "text": "Bonjour"},
            {"start": 1, "end": 2, "text": "Sous-titrage ST' 501"},      # hallucination connue
            {"start": 2, "end": 3, "text": "Amen"}, {"start": 3, "end": 4, "text": "Amen"}, {"start": 4, "end": 5, "text": "Amen"},
            {"start": 5, "end": 6, "text": "Fin", "words": [{"w": "Fin", "s": 5, "e": 6}]}]
    out = transcription.clean_segments(segs)
    assert [s["text"] for s in out] == ["Bonjour", "Amen", "Amen", "Fin"]     # la 3e répétition identique est retirée
    assert out[-1]["words"][0]["w"] == "Fin"


# ------------------------- frontières -------------------------
def test_make_blocks():
    units = [{"start": t, "end": t + 5, "text": f"u{t}"} for t in (0, 10, 25, 31, 70)]
    blocks = boundaries.make_blocks(units, 30)
    assert [(b["a"], b["b"]) for b in blocks] == [(0, 3), (3, 4), (4, 5)]
    assert blocks[0]["text"] == "u0 u10 u25"


def _two_topics(n_each=12):
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    return np.array([e1] * n_each + [e2] * n_each)


def test_depth_peaks_at_topic_change():
    emb = _two_topics()
    depth = boundaries.depth_scores(emb, np.ones(len(emb)), window=4)
    assert int(np.argmax(depth)) in (12, 11, 13)        # creux de similarité autour du changement de sujet
    assert depth[12] > 10 * np.median(depth[1:5])        # bien plus profond qu'au milieu d'un sujet
    assert depth.shape == (24,) and (depth >= 0).all()


def test_pick_boundaries_respects_min_gap_and_picks_the_dip():
    emb = _two_topics(20)
    depth = boundaries.depth_scores(emb, np.ones(len(emb)), window=4)
    btimes = np.arange(len(emb)) * 30.0                  # blocs de 30 s : 20 minutes au total
    picked = boundaries.pick_boundaries(depth, btimes, 0.0, 1200.0, min_gap=300, target=600, max_len=1200, strictness=0.0)
    assert len(picked) == 1 and abs(btimes[picked[0]] - 600) <= 60
    # sections trop longues : coupure forcée même sans creux net
    flat = np.zeros(80)
    forced = boundaries.pick_boundaries(flat, np.arange(80) * 30.0, 0.0, 2400.0, min_gap=300, target=2400, max_len=1200, strictness=0.0)
    assert len(forced) >= 1


def test_snap_to_pause_prefers_long_gap_after_sentence_end():
    units = [{"start": 0, "end": 10, "text": "a b."}, {"start": 10.2, "end": 20, "text": "c d"},
             {"start": 24, "end": 30, "text": "e f"}, {"start": 30.1, "end": 40, "text": "g"}]
    starts = [u["start"] for u in units]
    j = boundaries.snap_to_pause(units, starts, 22.0, radius=10)
    assert j == 2                                        # pause de 4 s avant l'unité 2


def test_section_dict():
    units = [{"start": 1, "end": 2, "text": "a"}, {"start": 2, "end": 9, "text": "b"}]
    assert boundaries.section_dict(units, 0, 2) == {"a": 0, "b": 2, "start": 1, "end": 9}


# ------------------------- thèmes -------------------------
def test_parse_llm_json_variants():
    assert themes.parse_llm_json('{"theme": "a", "description": "b", "suite_de": 2}')["suite_de"] == 2
    assert themes.parse_llm_json('blabla {"theme": "x", "description": "z"} fin')["theme"] == "x"
    cut = themes.parse_llm_json('"theme": "Un thème", "description": "Une desc tronquée')
    assert cut["theme"] == "Un thème" and cut["suite_de"] is None
    assert themes.parse_llm_json("n'importe quoi") is None


def test_clean_text():
    assert themes.clean_text('  «bonjour   le monde» ') == "Bonjour le monde"
    assert themes.clean_text(None) == ""


def test_build_prompt_contains_context_and_glossary():
    cfg = Config()
    p = themes.build_prompt("texte", ["Thème un", "Thème deux"], 3, 10, 8, cfg)
    assert "1. Thème un" in p and "2. Thème deux" in p and "n°3 sur environ 10" in p
    assert '"suite_de" : 2 UNIQUEMENT' in p and cfg.GLOSSARY in p
    assert "null (première section)" in themes.build_prompt("x", [], 1, 5, 8, cfg)


def _sections():
    base = {"start": 0, "end": 60, "description": "d", "short_title": ""}
    return [{**base, "theme": "Un", "part_of": None, "short_title": "Titre un"},
            {**base, "theme": "Deux", "part_of": 0, "short_title": "Titre deux"},
            {**base, "theme": "Trois", "part_of": None}]


def test_finalize_titles_chains_parts():
    s = themes.finalize_titles(_sections(), Config())
    assert [r["id"] for r in s] == [1, 2, 3]
    assert s[0]["theme_display"] == "Un (partie 1/2)" and s[1]["theme_display"] == "Deux (partie 2/2)"
    assert s[0]["chapter_title"] == "Titre un (1/2)" and s[2]["theme_display"] == "Trois"
    assert s[2]["chapter_title"] == "Trois"             # sans titre court : le thème sert de titre


def test_short_title_truncates_on_word_boundary():
    cfg = Config(CHAPTER_TITLE_MAX=20)
    t = themes.short_title_of({"short_title": "Un titre beaucoup trop long pour tenir", "theme": "x"}, cfg)
    assert len(t) <= 20 and t.endswith("…") and " " not in t[-3:]


def test_build_plan_format():
    s = themes.finalize_titles(_sections(), Config())
    plan = themes.build_plan(s, long_fmt=False)
    assert plan.startswith("I. Thème : Un (partie 1/2)\nSection : [00:00 - 01:00]\nDescription : d\n")
    assert "III. Thème : Trois" in plan


# ------------------------- vidéo -------------------------
def test_video_bounds_are_contiguous():
    secs = [{"start": 5}, {"start": 100}, {"start": 250}]
    assert video.video_bounds(secs, 400) == [(0.0, 100.0), (100.0, 250.0), (250.0, 400.0)]


def test_yt_ts_and_ffmeta_escape():
    assert [video.yt_ts(t) for t in (5, 65, 3700)] == ["0:05", "1:05", "1:01:40"]
    assert video.ffmeta_escape("a=b;c#d\\e\nf") == "a\\=b\\;c\\#d\\\\e f"


# ------------------------- projets, playlists -------------------------
def _project(name):
    return Project(source=Path(name), dir=Path("/tmp/x"), duration=10, has_video=True)


def test_for_each_continues_after_failure_for_playlists():
    projects = [_project("a.mp4"), _project("b.mp4"), _project("c.mp4")]
    done = []

    def step(p):
        if p.source.name == "b.mp4":
            raise RuntimeError("boum")
        done.append(p.source.name)

    for_each(projects, "test", step)
    assert done == ["a.mp4", "c.mp4"] and projects[1].error.startswith("test : boum") and projects[0].error is None


def test_for_each_raises_for_single_video():
    with pytest.raises(RuntimeError):
        for_each([_project("a.mp4")], "test", lambda p: (_ for _ in ()).throw(RuntimeError("boum")))


def test_for_each_skips_failed_projects_in_next_steps():
    projects = [_project("a.mp4"), _project("b.mp4")]
    projects[0].error = "avant"
    seen = []
    for_each(projects, "suite", lambda p: seen.append(p.source.name))
    assert seen == ["b.mp4"]


def test_project_paths_and_long_format():
    p = Project(source=Path("v.mp4"), dir=Path("/r/v"), duration=3600, has_video=True)
    assert p.transcript_json == Path("/r/v/transcript.json") and p.sections_json.name == "sections.json" and p.long_fmt
    assert not Project(source=Path("v.mp4"), dir=Path("/r"), duration=3599, has_video=True).long_fmt


def test_group_of_and_list_videos(tmp_path):
    data = tmp_path / "data"
    (data / "Ma playlist").mkdir(parents=True)
    (data / "Ma playlist" / "2 - b.mp4").write_bytes(b"")
    (data / "Ma playlist" / "1 - a.mp4").write_bytes(b"")
    (data / "seule.mp4").write_bytes(b"")
    (data / "Ma playlist" / "sections_video").mkdir()
    (data / "Ma playlist" / "sections_video" / "01_x.mp4").write_bytes(b"")
    (data / "notes.txt").write_text("x")
    vids = list_videos(data)
    assert [v.name for v in vids] == ["1 - a.mp4", "2 - b.mp4", "seule.mp4"]      # triées, sans sections_video ni .txt
    assert _group_of(data / "Ma playlist" / "1 - a.mp4", data) == "Ma playlist"
    assert _group_of(data / "seule.mp4", data) is None
