"""Un `Project` = une vidéo à traiter, avec ses fichiers de résultats et l'état produit par chaque étape."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Project:
    source: Path                       # vidéo (ou audio) d'origine
    dir: Path                          # dossier des résultats de cette vidéo
    duration: float
    has_video: bool
    group: str | None = None           # nom de la playlist / du dossier d'origine, None pour un fichier isolé
    units: list | None = None          # segments de la transcription (étape 4)
    transcript: dict | None = None
    sections: list = field(default_factory=list)   # sections thématiques (étape 5)
    error: str | None = None           # message si une étape a échoué (le traitement continue avec les autres vidéos)

    @property
    def title(self):
        return self.source.name

    @property
    def audio(self):
        return self.dir / "audio.wav"

    @property
    def transcript_json(self):
        return self.dir / "transcript.json"

    @property
    def sections_json(self):
        return self.dir / "sections.json"

    @property
    def plan_txt(self):
        return self.dir / "plan_thematique.txt"

    @property
    def long_fmt(self):
        return self.duration >= 3600   # repères [H:MM:SS] pour les vidéos d'une heure ou plus


def for_each(projects, step_name, fn):
    """Applique `fn(project)` à chaque projet, avec un en-tête [i/n] lorsqu'il y en a plusieurs.

    Pour un seul projet, les erreurs remontent telles quelles. Pour une playlist, l'échec d'une vidéo
    est noté dans `project.error` et les suivantes sont traitées.
    """
    n = len(projects)
    for i, p in enumerate(projects, 1):
        if p.error:
            continue
        if n > 1:
            print(f"\n=== [{i}/{n}] {p.title} : {step_name} ===")
        try:
            fn(p)
        except Exception as e:
            if n == 1:
                raise
            p.error = f"{step_name} : {str(e)[:300]}"
            print("  ÉCHEC :", p.error)
