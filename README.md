# socialMediaContents

Outils pour transformer de longues vidéos d'enseignement (prédications, conventions, réunions en français) en contenus pour les réseaux sociaux : **YouTube d'abord**, puis Instagram et TikTok.

> **État actuel : phase 1 terminée.** Le pipeline découpe une vidéo longue en **chapitres thématiques** de 5 à 20 minutes, avec un titre, un thème et une description pour chacun. Il produit les chapitres YouTube et un fichier vidéo par section. La détection d'extraits courts (Shorts, Reels, TikTok) est l'étape suivante (voir [Feuille de route](#feuille-de-route)).

---

## Ce que fait le pipeline

Le notebook [`decoupage_thematique.ipynb`](decoupage_thematique.ipynb) enchaîne quatre étapes :

1. **Transcription** horodatée (au mot près) avec [faster-whisper], ou import d'un sous-titre `.vtt` / `.srt` existant (Teams, Zoom).
2. **Détection des changements de sujet** : le texte est découpé en blocs de 30 s, puis vectorisé (`BAAI/bge-m3`). On compare ensuite les 3 minutes avant et après chaque point de coupure possible. Les creux de similarité deviennent des frontières, recalées sur la pause ou la fin de phrase la plus proche.
3. **Rédaction par un LLM local** (Qwen2.5 7B via Ollama) : pour chaque section, le modèle rédige un thème en une phrase, un titre court de chapitre et une description. Il indique aussi si la section prolonge la précédente. Dans ce cas, les deux sont fusionnées ou numérotées « 1/2, 2/2 ». Un garde-fou par similarité des thèmes (seuil 0,88) rattrape les suites que le LLM n'a pas signalées.
4. **Découpage vidéo** avec ffmpeg : un fichier par section, une copie de la vidéo complète avec chapitres intégrés et la liste des chapitres à coller dans la description YouTube.

Chaque étape écrit son résultat sur le disque et **est sautée si ce résultat existe déjà**. Pour en relancer une, utilise `FORCE = {"transcribe"}`, `{"themes"}` ou `{"video"}`.

## Installation (PC local, sans GPU)

Configuration de référence : Linux, CPU seul, Anaconda (Python 3.13).

```bash
# 1. ffmpeg
sudo apt install ffmpeg

# 2. Dépendances Python (le notebook installe aussi ce qui manque si INSTALL_DEPS = True)
pip install faster-whisper sentence-transformers soundfile numpy requests tqdm yt-dlp

# 3. LLM local : Ollama + Qwen2.5 7B avec un contexte de 12k tokens
ollama pull qwen2.5:7b-instruct
ollama create qwen2.5-7b-12k -f Modelfile
```

Le [`Modelfile`](Modelfile) augmente le contexte à 12 288 tokens. Avec les 4k par défaut d'Ollama, les sections longues (environ 20 000 caractères) seraient tronquées.

Le notebook fonctionne aussi sur **Google Colab (GPU T4)** : Whisper `large-v3-turbo` et Qwen2.5-7B en 4 bits via `transformers`. Les backends `openai` (tout serveur compatible) et `anthropic` sont possibles, mais **le texte est alors envoyé à un tiers**.

## Utilisation

1. Ouvre `decoupage_thematique.ipynb` et modifie **uniquement la cellule « Configuration »** :
   - `INPUT` : chemin d'une vidéo (par ex. `data/ma_video.mp4`) ou URL YouTube ;
   - `INITIAL_PROMPT` / `GLOSSARY` : noms propres et jargon à bien orthographier (Jésus-Christ, CMCI, CMFI, ZTF…) ;
   - `TARGET_SECTION_SEC`, `MIN_SECTION_SEC`, `MAX_SECTION_SEC` : longueur des sections (8 min visées par défaut, entre 5 et 20 min).
2. Exécute toutes les cellules.
3. Les résultats arrivent dans `outputs/<nom-de-la-video>/`.

Compte plusieurs heures de calcul sur CPU pour une vidéo de 2 à 3 h avec Whisper `medium`. `WHISPER_MODEL = "small"` va environ 2 à 3 fois plus vite, mais est moins fiable.

## Fichiers produits

| Fichier | Contenu |
| --- | --- |
| `transcript.json` | Transcription horodatée (segments et mots) |
| `sections.json` | Sections : bornes, thème, titre court, description, paramètres utilisés |
| `plan_thematique.txt` | Plan lisible : `I. Thème : … / Section : [début - fin] / Description : …` |
| `transcription_structuree.md` | Plan, puis texte complet de chaque section avec repères de temps |
| `chapitres_youtube.txt` | Chapitres à coller dans la description YouTube |
| `chapitres.ffmetadata` | Chapitres au format ffmpeg |
| `<nom>_chapitres.mp4` | Vidéo complète avec chapitres intégrés (sans ré-encodage) |
| `sections_video/NN_<titre>.mp4` | Un fichier vidéo par section (+ `manifest.json`) |
| `audio.wav` | Audio mono 16 kHz extrait (cache) |

## Structure du dépôt

```text
.
├── decoupage_thematique.ipynb   # le pipeline complet
├── Modelfile                    # modèle Ollama qwen2.5-7b-12k
├── data/                        # vidéos sources (non versionnées)
└── outputs/
    ├── downloads/               # vidéos téléchargées depuis YouTube (non versionnées)
    ├── cpj2026-day1/            # exemple : convention CPJ 2026, jour 1 (3 h 28, 19 chapitres)
    └── v-18dl8rj9s/             # exemple : enseignement sur l'appel de Dieu (1 h 51, 11 chapitres)
```

Seuls les **résultats texte** (JSON, TXT, MD) sont versionnés. Les vidéos et l'audio (`*.mp4`, `*.wav`, etc.), soit plusieurs Go, sont exclus par le [`.gitignore`](.gitignore).

> Note : l'exemple `cpj2026-day1` a été produit **avant** les corrections de la phase 1 (Whisper `small`, titres de chapitres tronqués). `v-18dl8rj9s` reflète la version actuelle (Whisper `medium`, horodatage par mot).

## Feuille de route

- [x] **Phase 1 : chapitres fiables.** Un titre propre à chaque chapitre, des titres courts (≤ 50 caractères), une détection stricte des suites avec garde-fou par similarité, Whisper `medium` et l'horodatage au mot.
- [ ] **Phase 2 : extraits pour les réseaux.** Détection de passages forts de 30 à 90 s, page de relecture humaine, rendu vertical 9:16 avec sous-titres animés mot à mot.
- [ ] **Phase 3 : industrialisation.** Sortir le code du notebook (modules Python, ligne de commande) et publier automatiquement via l'API YouTube.
- [ ] **Phase 4 : diffusion élargie.** Compilations, statistiques, Instagram et TikTok.

## Limites connues

- Les frontières sont détectées automatiquement : **relis le plan** avant de publier.
- Un LLM de 7B peut mal reformuler une idée ou rater une fusion. Pour de meilleurs résultats, utilise un modèle plus gros.
- La qualité dépend de la source : bruit, accents, mélange de langues. Les vidéos en 360p limitent la qualité des futurs extraits verticaux.

## Cadre légal

Enregistrer et republier un enseignement ou une réunion demande l'accord des personnes concernées. Respecte aussi les conditions des plateformes et le droit d'auteur, notamment pour la musique de fond.

[faster-whisper]: https://github.com/SYSTRAN/faster-whisper
