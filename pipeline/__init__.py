"""Pipeline de découpage thématique : transcription, sections, thèmes (LLM), découpe vidéo.

Chaque étape du notebook `decoupage_thematique.ipynb` appelle une fonction de ce paquet :

| Étape                  | Module                      | Fonction                |
|------------------------|-----------------------------|-------------------------|
| Configuration          | pipeline.config             | Config                  |
| Environnement          | pipeline.environment        | setup_environment       |
| Source (fichier / URL) | pipeline.source             | resolve_sources         |
| Transcription          | pipeline.transcription      | transcribe              |
| Sections et thèmes     | pipeline.themes             | build_sections          |
| Texte complet          | pipeline.exports            | export_full_text        |
| Découpe vidéo          | pipeline.video              | cut_video               |
| Résultats              | pipeline.exports            | show_results            |

Les modules `config`, `utils`, `environment`, `project` et `download` n'importent que la bibliothèque
standard : `setup_environment` peut donc installer les paquets manquants (numpy, faster-whisper...) avant
que les autres modules ne soient importés.
"""
