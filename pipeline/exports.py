"""Exports texte (transcription structurée) et résumé final des résultats."""
import shutil

from .project import for_each
from .utils import fmt_clock, fmt_time, to_roman


def export_full_text_project(project):
    units, sections, long_fmt = project.units, project.sections, project.long_fmt
    lines = [f"# Transcription structurée — {project.source.name}", "", f"Durée : {fmt_time(project.duration)}", "", "## Plan", ""]
    for r in sections:
        lines.append(f"- **{to_roman(r['id'])}.** {r['theme_display']} `[{fmt_clock(r['start'], long_fmt)} - {fmt_clock(r['end'], long_fmt)}]`")
    for r in sections:
        lines += ["", f"## {to_roman(r['id'])}. {r['theme_display']}", "",
                  f"> **Section** : {fmt_clock(r['start'], long_fmt)} - {fmt_clock(r['end'], long_fmt)}  ", f"> **Description** : {r['description']}", ""]
        para = []
        for u in units[r["a"]:r["b"]]:
            para.append(u["text"])
            if len(para) >= 6:
                lines.append(f"`[{fmt_clock(u['start'], long_fmt)}]` " + " ".join(para) + "\n")
                para = []
        if para:
            lines.append(f"`[{fmt_clock(units[r['b'] - 1]['start'], long_fmt)}]` " + " ".join(para) + "\n")
    (project.dir / "transcription_structuree.md").write_text("\n".join(lines), encoding="utf-8")
    print("Exporté :", project.dir / "transcription_structuree.md")


def export_full_text(cfg, projects):
    """Étape 6 (optionnelle) : écrit `transcription_structuree.md` pour chaque vidéo."""
    if not cfg.EXPORT_FULL_TEXT:
        return
    for_each([p for p in projects if p.sections], "texte complet", export_full_text_project)


def show_results(cfg, env, projects):
    """Liste les fichiers produits ; signale les vidéos en échec ; sur Colab sans Drive, prépare un zip à télécharger."""
    for p in projects:
        print("Résultats dans :", p.dir)
        for f in sorted(p.dir.iterdir()):
            print("  ", f.name + ("/" if f.is_dir() else ""))

        # Colab sans Drive : récupérer l'essentiel en un zip
        if env.in_colab and not cfg.MOUNT_DRIVE:
            from google.colab import files
            keep = [f for f in p.dir.iterdir() if f.suffix in (".txt", ".json", ".md", ".ffmetadata")]
            tmp = p.dir / "resultats"
            tmp.mkdir(exist_ok=True)
            for f in keep:
                shutil.copy(f, tmp / f.name)
            files.download(shutil.make_archive(str(p.dir / "resultats"), "zip", tmp))

    failed = [p for p in projects if p.error]
    if failed:
        print(f"\n{len(failed)} vidéo(s) en échec sur {len(projects)} :")
        for p in failed:
            print(f"  - {p.title} : {p.error}")
