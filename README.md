# socialMediaContents

Outils pour transformer de longues vidéos d'enseignement (prédications, conventions, réunions en français) en contenus pour les réseaux sociaux : **YouTube d'abord**, puis Instagram et TikTok.

> **État actuel : phase 1 terminée.** Le pipeline découpe une vidéo longue en **chapitres thématiques** de 5 à 20 minutes, avec un titre, un thème et une description pour chacun. Il produit les chapitres YouTube et un fichier vidéo par section. La détection d'extraits courts (Shorts, Reels, TikTok) est l'étape suivante (voir [Feuille de route](#feuille-de-route)).

---

## Ce que fait le pipeline

[www.youtube.com/watch?v=eC26CBR310E&amp;list=PLtWYbjRdhgw4ob1BG4mqOBaz-TFf5osn-&amp;index=2](https://www.youtube.com/watch?v=eC26CBR310E&list=PLtWYbjRdhgw4ob1BG4mqOBaz-TFf5osn-&index=2)Le notebook [`decoupage_thematique.ipynb`](decoupage_thematique.ipynb) enchaîne quatre étapes. Il ne contient que la configuration et un appel par étape : le code est rangé par composant dans le dossier [`pipeline/`](pipeline/) (voir [Structure du dépôt](#structure-du-dépôt)).

1. **Transcription** horodatée (au mot près) avec [faster-whisper][faster-whisper], ou import d'un sous-titre `.vtt` / `.srt` existant (Teams, Zoom).
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

Le notebook fonctionne aussi sur **Google Colab (GPU T4)** : Whisper `large-v3-turbo` et Qwen2.5-7B en 4 bits via `transformers`. Sur Colab, la deuxième cellule récupère le code par `git clone` : le dossier `pipeline/` doit donc être **poussé sur GitHub**. Les backends `openai` (tout serveur compatible) et `anthropic` sont possibles, mais **le texte est alors envoyé à un tiers**.

## Utilisation

1. Ouvre `decoupage_thematique.ipynb` et modifie **uniquement la cellule « Configuration »** :
   - `INPUT` : un fichier (par ex. `data/ma_video.mp4`), une **URL YouTube** (vidéo ou playlist) ou un **dossier de vidéos** ;
   - `INITIAL_PROMPT` / `GLOSSARY` : noms propres et jargon à bien orthographier (Jésus-Christ, CMCI, CMFI, ZTF…) ;
   - `TARGET_SECTION_SEC`, `MIN_SECTION_SEC`, `MAX_SECTION_SEC` : longueur des sections (8 min visées par défaut, entre 5 et 20 min).
2. Exécute toutes les cellules.
3. Les résultats arrivent dans `outputs/<nom-de-la-video>/`.

Compte plusieurs heures de calcul sur CPU pour une vidéo de 2 à 3 h avec Whisper `medium`. `WHISPER_MODEL = "small"` va environ 2 à 3 fois plus vite, mais est moins fiable.

### Vidéos YouTube et playlists

Une URL dans `INPUT` est téléchargée avec `yt-dlp` dans le dossier **`data/`** (réglage `DOWNLOAD_DIR`), en 1080p au maximum (sinon le format inférieur le plus proche), avec des fichiers nommés d'après leur titre :

| Cas | Fichier téléchargé | Résultats |
| --- | --- | --- |
| Vidéo seule | `data/<titre>.mp4` | `outputs/<titre>/` |
| Playlist | `data/<titre de la playlist>/<NN> - <titre>.mp4` | `outputs/<playlist>/<NN-titre>/` (un dossier complet par vidéo) |

- Une URL `watch?v=…&list=…` ne télécharge que la vidéo. Pour toute la playlist, mets `PLAYLIST = True`. Une URL `/playlist?list=…` est toujours traitée comme une playlist.
- `PLAYLIST_ITEMS = "1-3"` limite aux premiers éléments.
- `INPUT = "data/<titre de la playlist>"` (un dossier) retraite une playlist déjà téléchargée sans la retélécharger.
- Une vidéo en échec (privée, supprimée, erreur de calcul) n'arrête pas les suivantes : elle est signalée dans le résumé final.
- `TRANSCRIPT_FILE` (sous-titre `.vtt`) ne s'applique qu'à une seule vidéo.

Pour **seulement télécharger**, sans transcrire :

```bash
python Scripts/ytbeDwnlder.py "<url>"                     # vidéo ou playlist → data/
python Scripts/ytbeDwnlder.py "<url>" --items 1-5         # certains éléments d'une playlist
python Scripts/ytbeDwnlder.py "<watch?v=…&list=…>" --playlist
python Scripts/ytbeDwnlder.py "<url>" --audio -o ~/Musique # mp3 seul, autre dossier
```

## Fichiers produits

| Fichier                           | Contenu                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------- |
| `transcript.json`               | Transcription horodatée (segments et mots)                                     |
| `sections.json`                 | Sections : bornes, thème, titre court, description, paramètres utilisés      |
| `plan_thematique.txt`           | Plan lisible : `I. Thème : … / Section : [début - fin] / Description : …` |
| `transcription_structuree.md`   | Plan, puis texte complet de chaque section avec repères de temps               |
| `chapitres_youtube.txt`         | Chapitres à coller dans la description YouTube                                 |
| `chapitres.ffmetadata`          | Chapitres au format ffmpeg                                                      |
| `<nom>_chapitres.mp4`           | Vidéo complète avec chapitres intégrés (sans ré-encodage)                  |
| `sections_video/NN_<titre>.mp4` | Un fichier vidéo par section (+ `manifest.json`)                             |
| `audio.wav`                     | Audio mono 16 kHz extrait (cache)                                               |

## Structure du dépôt

```text
.
├── decoupage_thematique.ipynb   # configuration + un appel par étape
├── pipeline/                    # le code, un module par composant
│   ├── config.py                #   réglages (classe Config)
│   ├── utils.py                 #   fonctions générales (temps, noms de fichiers, ffmpeg...)
│   ├── environment.py           #   paquets, ffmpeg, GPU/CPU, dossiers de travail
│   ├── download.py              #   téléchargement yt-dlp (vidéo et playlist)
│   ├── source.py                #   fichier / dossier / URL → liste de vidéos à traiter
│   ├── project.py               #   une vidéo = un Project ; traitement par lot tolérant aux échecs
│   ├── transcription.py         #   faster-whisper, import .vtt/.srt
│   ├── boundaries.py            #   blocs, similarité, creux, frontières, calage sur les pauses
│   ├── llm.py                   #   moteurs de rédaction (transformers, Ollama/OpenAI, Anthropic)
│   ├── themes.py                #   thèmes, titres, descriptions, fusion des suites, plan
│   ├── exports.py               #   texte structuré, résumé des résultats
│   └── video.py                 #   chapitres, découpe par section, vidéo avec chapitres
├── Scripts/ytbeDwnlder.py       # télécharger une vidéo ou une playlist en ligne de commande
├── tests/                       # tests des fonctions pures (pytest)
├── docs/fondements_mathematiques.md   # les mathématiques derrière chaque brique, et comment ajuster
├── Modelfile                    # modèle Ollama qwen2.5-7b-12k
├── data/                        # vidéos téléchargées ou sources (non versionnées)
└── outputs/
    ├── cpj2026-day1/            # exemple : convention CPJ 2026, jour 1 (3 h 28, 19 chapitres)
    └── v-18dl8rj9s/             # exemple : enseignement sur l'appel de Dieu (1 h 51, 11 chapitres)
```

Pour lancer les tests : `python -m pytest tests`. Pour comprendre les formules derrière les réglages (fenêtre de comparaison, seuil de similarité, température du LLM, CRF vidéo…), voir [`docs/fondements_mathematiques.md`](docs/fondements_mathematiques.md).

Seuls les **résultats texte** (JSON, TXT, MD) sont versionnés. Les vidéos et l'audio (`*.mp4`, `*.wav`, etc.), soit plusieurs Go, sont exclus par le [`.gitignore`](.gitignore).

> Note : l'exemple `cpj2026-day1` a été produit **avant** les corrections de la phase 1 (Whisper `small`, titres de chapitres tronqués). `v-18dl8rj9s` reflète la version actuelle (Whisper `medium`, horodatage par mot).

## Feuille de route

- [X] **Phase 1 : chapitres fiables.** Un titre propre à chaque chapitre, des titres courts (≤ 50 caractères), une détection stricte des suites avec garde-fou par similarité, Whisper `medium` et l'horodatage au mot.
- [ ] **Phase 2 : extraits pour les réseaux.** Détection de passages forts de 30 à 90 s, page de relecture humaine, rendu vertical 9:16 avec sous-titres animés mot à mot.
- [ ] **Phase 3 : industrialisation.** ~~Sortir le code du notebook en modules Python~~ (fait : dossier `pipeline/`, playlists prises en charge). Reste : une ligne de commande complète et la publication automatique via l'API YouTube.
- [ ] **Phase 4 : diffusion élargie.** Compilations, statistiques, Instagram et TikTok.

## Limites connues

- Les frontières sont détectées automatiquement : **relis le plan** avant de publier.
- Un LLM de 7B peut mal reformuler une idée ou rater une fusion. Pour de meilleurs résultats, utilise un modèle plus gros.
- La qualité dépend de la source : bruit, accents, mélange de langues. Les vidéos en 360p limitent la qualité des futurs extraits verticaux.

## Cadre légal

Enregistrer et republier un enseignement ou une réunion demande l'accord des personnes concernées. Respecte aussi les conditions des plateformes et le droit d'auteur, notamment pour la musique de fond.

[faster-whisper]: https://github.com/SYSTRAN/faster-whisper
