"""Réglages du pipeline : une seule classe, les mêmes noms que dans la cellule « Configuration » du notebook.

Toute faute de frappe sur un nom de réglage lève une erreur (`Config(TARGET_SECTION_SECS=...)` → TypeError).
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Entrée ---------------------------------------------------------------
    INPUT: str = ""                  # fichier, dossier de vidéos, URL YouTube (vidéo ou playlist), ou "" (Colab : téléversement)
    TRANSCRIPT_FILE: str = ""        # optionnel : .vtt / .srt Teams, Zoom... -> remplace Whisper (une seule vidéo)
    PLAYLIST: bool = False           # URL watch?v=..&list=.. : True = toute la playlist (une URL /playlist?list=.. en est toujours une)
    PLAYLIST_ITEMS: str = ""         # éléments de la playlist, ex. "1-3" ou "1,3,7" ; vide = tous

    # --- Étapes ---------------------------------------------------------------
    RUN_TRANSCRIBE: bool = True
    RUN_THEMES: bool = True
    RUN_VIDEO: bool = True
    FORCE: set = field(default_factory=set)   # {"audio", "transcribe", "themes", "video"}
    INSTALL_DEPS: bool = True

    # --- Dossiers -------------------------------------------------------------
    WORK_DIR: str = "outputs"
    DOWNLOAD_DIR: str = "data"       # vidéos téléchargées depuis YouTube (Colab : <dossier de travail>/data)
    MOUNT_DRIVE: bool = True
    DRIVE_SUBDIR: str = "meeting_pipeline"

    # --- Transcription --------------------------------------------------------
    LANGUAGE: str = "fr"
    MULTILINGUAL: bool = False
    WHISPER_MODEL: str = "auto"
    WORD_TIMESTAMPS: bool = True
    INITIAL_PROMPT: str = "Jésus-Christ, Saint-Esprit, CMCI, CMFI, ZTF, convention, jeûne, réveil, repentance."

    # --- Découpage thématique -------------------------------------------------
    MIN_SECTION_SEC: int = 300
    TARGET_SECTION_SEC: int = 480
    MAX_SECTION_SEC: int = 1200
    BLOCK_SEC: int = 30
    TOPIC_WINDOW_SEC: int = 180
    TOPIC_STRICTNESS: float = 0.0
    EMBED_MODEL: str = "BAAI/bge-m3"

    # --- Rédaction des thèmes et descriptions (LLM) --------------------------
    LLM_BACKEND: str = "auto"
    OUTPUT_LANGUAGE: str = "français"
    GLOSSARY: str = "Jésus-Christ, Saint-Esprit, CMCI, CMFI"
    CHAPTER_TITLE_MAX: int = 50
    MERGE_CONTINUING: bool = True
    THEME_SIM_CONTINUE: float = 0.88
    LLM_MAX_CHARS: int = 20000

    LLM_HF_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    LLM_4BIT: bool = True
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "qwen2.5-7b-12k"
    LLM_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-5-5"
    ANTHROPIC_URL: str = "https://api.anthropic.com/v1/messages"

    # --- Exports ---------------------------------------------------------------
    EXPORT_FULL_TEXT: bool = True
    CUT_SECTIONS: bool = True
    CUT_MODE: str = "reencode"       # "reencode" (précis, plus lent) | "copy" (rapide, précis à quelques secondes près)
    EMBED_CHAPTERS: bool = True

    def __post_init__(self):
        self.FORCE = set(self.FORCE)
