#!/usr/bin/env python3
"""Télécharge une vidéo ou une playlist YouTube avec yt-dlp, en nommant les fichiers d'après leur titre.

Exemples :
    python Scripts/ytbeDwnlder.py "https://www.youtube.com/watch?v=V_18dl8rJ9s"
    python Scripts/ytbeDwnlder.py "https://www.youtube.com/playlist?list=PL..."          # toute la playlist
    python Scripts/ytbeDwnlder.py "https://www.youtube.com/playlist?list=PL..." --items 1-3
    python Scripts/ytbeDwnlder.py "https://www.youtube.com/watch?v=X&list=PL..." --playlist
    python Scripts/ytbeDwnlder.py "<url>" --audio -o ~/Musique

Nommage :
    vidéo seule : <dossier>/<titre>.mp4
    playlist    : <dossier>/<titre de la playlist>/01 - <titre>.mp4
"""
import argparse
import sys
from pathlib import Path

# La logique de téléchargement est partagée avec le notebook : voir pipeline/download.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.download import download_url  # noqa: E402

# Dossier de sortie par défaut : data/ à la racine du projet (indépendant du dossier de lancement).
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    p = argparse.ArgumentParser(description="Télécharge une vidéo ou une playlist YouTube (fichiers nommés d'après le titre).")
    p.add_argument("url", help="URL d'une vidéo ou d'une playlist YouTube")
    p.add_argument("-o", "--out-dir", default=str(DEFAULT_OUT_DIR), help="dossier de sortie (défaut : data/ du projet)")
    p.add_argument("--playlist", action="store_true",
                   help="si l'URL est une vidéo appartenant à une playlist (watch?v=..&list=..), télécharger toute la playlist")
    p.add_argument("--items", help="éléments de la playlist à télécharger, ex. 1-5 ou 1,3,7 (défaut : tous)")
    p.add_argument("--audio", action="store_true", help="extraire l'audio seul en mp3")
    p.add_argument("--max-height", type=int, default=1080, help="hauteur vidéo maximale (défaut : 1080)")
    args = p.parse_args()

    try:
        paths, failed = download_url(args.url, args.out_dir, playlist=args.playlist, audio_only=args.audio,
                                     max_height=args.max_height, items=args.items)
    except Exception as e:
        sys.exit(f"Erreur : {e}")

    print(f"\n{len(paths)} fichier(s) prêt(s) :")
    for path in paths:
        print(f"  {path}")
    if failed:
        sys.exit("Certaines vidéos n'ont pas pu être téléchargées (privées, supprimées, bloquées...) : voir les erreurs ci-dessus.")


if __name__ == "__main__":
    main()
