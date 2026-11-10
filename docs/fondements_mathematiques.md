# Fondements mathématiques du pipeline

Ce document explique **les mathématiques derrière chaque brique** du notebook `decoupage_thematique.ipynb`, relie chaque formule à la **ligne de code** et au **paramètre** qui la porte, et propose une méthode pour **s'en imprégner et ajuster** les réglages en connaissance de cause.

Les formules sont en LaTeX (`$…$`). Elles s'affichent dans l'aperçu Markdown de VS Code et sur GitHub.

> **Ce qui est vérifié et ce qui ne l'est pas.** Les constantes et les valeurs chiffrées citées ont été relues dans le code ou mesurées sur ce dépôt (par exemple les 412 blocs de la convention CPJ 2026, ou le facteur de taille H.264 mesuré sur un extrait). Les chiffres issus de la littérature sont attribués à leur source. Ce qui est une règle empirique ou une approximation est signalé par *« environ »* ou *« règle empirique »*.

## Sommaire

0. [La carte : étape, mathématiques, paramètres](#0-la-carte)
1. [Notations et prérequis](#1-notations-et-prérequis)
2. [Du son aux nombres](#2-du-son-aux-nombres-échantillonnage-et-spectrogramme)
3. [Whisper : transcrire](#3-whisper--transcrire)
4. [Plongements de texte et similarité cosinus](#4-plongements-de-texte-et-similarité-cosinus)
5. [Détecter les changements de sujet](#5-détecter-les-changements-de-sujet)
6. [Le LLM : thèmes et descriptions](#6-le-llm--thèmes-et-descriptions)
7. [Fusion des suites et résumé sans LLM](#7-fusion-des-suites-et-résumé-sans-llm)
8. [La vidéo : H.264, images clés, découpe](#8-la-vidéo--h264-images-clés-découpe)
9. [S&#39;en imprégner et ajuster](#9-sen-imprégner-et-ajuster)

---

## 0. La carte

| Étape du notebook    | Idée mathématique                                        | Où dans le code                                 | Paramètres qui la gouvernent                                                             |
| --------------------- | ---------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Audio → tableau      | échantillonnage, théorème de Nyquist–Shannon           | `source.prepare_audio` (`-ar 16000 -ac 1`)   | —                                                                                        |
| Transcription         | spectrogramme log-mel, transformeur, recherche en faisceau | `transcription._whisper_pass`                  | `WHISPER_MODEL`, `LANGUAGE`, `INITIAL_PROMPT`, `WORD_TIMESTAMPS`                  |
| Vecteurs de texte     | plongement (embedding), similarité cosinus                | `boundaries.embed_texts`                       | `EMBED_MODEL`                                                                           |
| Frontières           | creux de similarité, profondeur, sélection gloutonne     | `boundaries.depth_scores`, `pick_boundaries` | `BLOCK_SEC`, `TOPIC_WINDOW_SEC`, `MIN/TARGET/MAX_SECTION_SEC`, `TOPIC_STRICTNESS` |
| Calage sur les pauses | optimisation d'un petit score local                        | `boundaries.snap_to_pause`                     | `BLOCK_SEC` (rayon)                                                                     |
| Thèmes, descriptions | modèle de langage autorégressif, softmax, température   | `llm.LLMClient`, `themes.build_prompt`       | `LLM_BACKEND`, `LLM_MAX_CHARS`, modèle                                               |
| Fusion des suites     | seuil sur un cosinus                                       | `themes.assemble_sections`                     | `THEME_SIM_CONTINUE`, `MERGE_CONTINUING`                                              |
| Repli sans LLM        | centralité autour du centroïde                           | `themes.extractive_describe`                   | —                                                                                        |
| Découpe vidéo       | quantification H.264 (CRF), images clés                   | `video.cut_project`                            | `CUT_MODE`                                                                              |

---

## 1. Notations et prérequis

Il suffit de maîtriser quatre objets.

**Vecteur et produit scalaire.** Un texte devient un vecteur $e \in \mathbb{R}^d$. Le produit scalaire de deux vecteurs est $a \cdot b = \sum_{k=1}^{d} a_k b_k$.

**Norme.** $\Vert a \Vert = \sqrt{a \cdot a}$. Un vecteur est dit *normalisé* si $\Vert a \Vert = 1$.

**Cosinus.** L'angle $\theta$ entre deux vecteurs vérifie

$$
\cos\theta \;=\; \frac{a \cdot b}{\Vert a \Vert \, \Vert b \Vert}\;\in[-1,\,1].
$$

Deux vecteurs de même direction donnent 1, deux vecteurs orthogonaux donnent 0. Si $a$ et $b$ sont normalisés, **le cosinus est simplement le produit scalaire**. C'est pourquoi le code demande `normalize_embeddings=True` puis utilise `np.dot` ou `@` (`boundaries.embed_texts`, `themes.assemble_sections`).

**Softmax.** Il transforme des scores réels $z_1,\dots,z_K$ en probabilités :

$$
\mathrm{softmax}(z)_i \;=\; \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}.
$$

On le retrouve dans l'attention (§3), la température du LLM (§6) et, indirectement, tout ce qui « choisit » un mot.

---

## 2. Du son aux nombres : échantillonnage et spectrogramme

### 2.1 Échantillonnage (Nyquist–Shannon)

Un signal sonore continu est mesuré $f_s$ fois par seconde. Pour représenter sans perte les fréquences jusqu'à $f_{\max}$, il faut

$$
f_s \;\ge\; 2\,f_{\max}.
$$

Le pipeline convertit tout en **mono, 16 kHz**, soit $f_{\max} = 8\ \text{kHz}$ : on garde l'essentiel des fréquences utiles à la parole, au prix d'un fichier beaucoup plus léger que de l'audio musical à 44,1 kHz.

**Conséquence mémoire, mesurable sur ce dépôt.** La convention CPJ 2026 dure 12 454 s. En 16 kHz, cela fait $12\,454 \times 16\,000 \approx 2\times10^{8}$ échantillons. En entier 16 bits (`pcm_s16le`), le fichier `audio.wav` pèse environ **400 Mo**. `transcription.run_whisper` le lit en `float32` (4 octets par échantillon), soit environ **800 Mo en RAM** pour 3 h 28. C'est un ordre de grandeur utile quand la machine manque de mémoire.

### 2.2 Spectrogramme log-mel

Whisper ne lit pas l'onde brute. Il la découpe en petites fenêtres, calcule leur spectre, puis le regroupe en bandes de fréquences perceptives.

1. **Transformée de Fourier à court terme (STFT).** On prend des fenêtres de $N$ échantillons, décalées de $h$ échantillons, et on calcule leur transformée de Fourier discrète
   $$
   X_t[k] \;=\; \sum_{n=0}^{N-1} x[n + t h]\; w[n]\; e^{-2\pi i\, kn/N}.
   $$

   Pour Whisper (constantes lues dans `faster_whisper.feature_extractor`) : $N = 400$ échantillons, soit **25 ms**, et $h = 160$ échantillons, soit **10 ms**. La résolution en fréquence vaut $\Delta f = f_s / N = 16\,000/400 = 40\ \text{Hz}$.
2. **Échelle mel.** L'oreille distingue mieux les graves que les aigus. L'échelle mel (forme classique) est
   $$
   m \;=\; 2595\,\log_{10}\!\Bigl(1 + \frac{f}{700}\Bigr).
   $$

   Whisper utilise la variante de la bibliothèque `librosa` (linéaire sous 1 kHz, logarithmique au-delà), qui a la même intention. Le spectre de puissance est regroupé en **80 bandes mel** (`feature_size = 80`).
3. **Logarithme.** On prend $\log(\text{énergie})$, ce qui rapproche la représentation de la perception du volume.

Pour une fenêtre de 30 s (`chunk_length = 30`), cela donne $30 / 0{,}01 = 3000$ images de 80 valeurs. Le codeur de Whisper les réduit par une convolution de pas 2 à **1 500 positions**, soit une position toutes les 20 ms (résultat de l'architecture décrite par Radford et al., 2022).

---

## 3. Whisper : transcrire

### 3.1 Un modèle qui estime une probabilité

Whisper est un transformeur **codeur-décodeur** entraîné sur 680 000 heures d'audio (Radford et al., 2022). Pour un audio $x$ et un texte $y = (y_1,\dots,y_T)$ (une suite de *jetons*, c'est-à-dire de morceaux de mots), il estime

$$
p(y \mid x) \;=\; \prod_{t=1}^{T} p\bigl(y_t \mid y_{<t},\, x\bigr).
$$

À chaque pas, le décodeur sort des scores sur tout le vocabulaire, que le softmax (§1) transforme en probabilités.

### 3.2 L'attention

Le cœur du transformeur (Vaswani et al., 2017). Pour des requêtes $Q$, des clés $K$ et des valeurs $V$ de dimension $d_k$ :

$$
\mathrm{Attention}(Q,K,V) \;=\; \mathrm{softmax}\!\Bigl(\frac{Q K^{\top}}{\sqrt{d_k}}\Bigr) V.
$$

Chaque position « regarde » toutes les autres avec un poids donné par le softmax d'un produit scalaire. La division par $\sqrt{d_k}$ évite que les scores ne grossissent avec la dimension et ne saturent le softmax. Le coût est **quadratique** en la longueur de la séquence : $O(n^2)$. Ce point revient au §6, car c'est la raison pour laquelle on limite le texte envoyé au LLM.

Le décodeur utilise aussi de l'**attention croisée** : ses requêtes viennent du texte déjà produit, ses clés et valeurs de l'audio encodé. C'est ce qui lui permet de « consulter » le son.

### 3.3 Décoder : la recherche en faisceau

Choisir à chaque pas le jeton le plus probable (décodage glouton) peut mener à une impasse. La **recherche en faisceau** (`beam_size = 5` dans le code) garde en parallèle les $B = 5$ meilleures hypothèses et vise à maximiser le score

$$
\text{score}(y) \;=\; \frac{1}{T}\sum_{t=1}^{T} \log p\bigl(y_t \mid y_{<t}, x\bigr)
$$

(divisé par la longueur pour ne pas favoriser les phrases très courtes ; c'est la forme par défaut de `faster-whisper`, avec une pénalité de longueur de 1). On travaille en logarithmes parce qu'un produit de nombreuses probabilités tombe vite sous la précision numérique, alors qu'une somme de logs reste stable.

Un faisceau plus large explore davantage mais coûte proportionnellement plus de calcul. Passer de `beam_size` 5 à 1 accélère, mais dégrade la fiabilité.

### 3.4 Détection de parole (VAD) et boucles de répétition

`vad_filter=True` lance un petit réseau (Silero VAD) qui donne pour chaque instant une probabilité de parole, comparée à un seuil. Les silences de plus de `min_silence_duration_ms=500` découpent l'audio, ce qui évite à Whisper d'« halluciner » du texte sur du bruit.

`condition_on_previous_text=False` empêche de réinjecter le texte précédent, ce qui limite les boucles de répétition. Le filtre `transcription.clean_segments` retire en plus les phrases parasites connues (expression régulière `HALLU`) et la troisième répétition identique.

### 3.5 Horodatage au mot : alignement par DTW

`WORD_TIMESTAMPS=True` demande à `faster-whisper` de situer chaque mot dans le temps. La méthode utilise les poids d'attention croisée de certaines « têtes d'alignement », puis un alignement par **déformation temporelle dynamique** (DTW, Sakoe et Chiba, 1978). On cherche le chemin qui apparie les $T$ jetons aux $M$ positions audio au moindre coût cumulé, avec la récurrence

$$
D(i,j) \;=\; c(i,j) + \min\bigl\{\,D(i-1,j),\; D(i,j-1),\; D(i-1,j-1)\,\bigr\}.
$$

### 3.6 Mesurer la qualité : le taux d'erreur de mots

La mesure standard d'une transcription est le **WER** (*word error rate*), comparé à une transcription de référence faite à la main :

$$
\mathrm{WER} \;=\; \frac{S + D + I}{N},
$$

où $S$ est le nombre de mots substitués, $D$ supprimés, $I$ insérés, et $N$ le nombre de mots de la référence (distance d'édition au niveau des mots). Pour choisir entre `small` et `medium`, c'est la seule mesure objective : transcris deux minutes avec chaque modèle, corrige l'une à la main, puis compte.

**Tailles de référence** (Radford et al., 2022) : `small` ≈ 244 M de paramètres, `medium` ≈ 769 M. Le temps de calcul sur CPU croît à peu près comme le nombre de paramètres, ce qui explique le rapport « environ 2 à 3 fois plus lent » du commentaire de `WHISPER_MODEL`.

---

## 4. Plongements de texte et similarité cosinus

Un **plongement** (*embedding*) associe à un texte un vecteur de $\mathbb{R}^d$ tel que **des textes de sens proche donnent des vecteurs de direction proche**. On mesure la proximité par le cosinus (§1).

Le modèle utilisé, `BAAI/bge-m3` (Chen et al., 2024), est un transformeur de type XLM-RoBERTa. Ses caractéristiques, relues dans sa configuration locale : **24 couches, 1024 dimensions, 16 têtes d'attention, vocabulaire de 250 002 jetons**, multilingue. La chaîne de traitement de `sentence-transformers` est : transformeur → *pooling* (résumé des vecteurs de jetons en un seul ; BGE-M3 s'appuie sur le jeton spécial `[CLS]`, selon sa documentation) → **normalisation L2** (division par la norme). Le vecteur final est donc de norme 1.

Le code limite la longueur des textes à 512 jetons (`max_seq_length = min(512, …)` dans `boundaries.get_embedder`). Un bloc de 30 s fait en moyenne **≈ 38 mots** (15 670 mots pour 412 blocs sur CPJ 2026), très en dessous de cette limite.

Deux idées à retenir :

- La **direction** d'un vecteur porte le sens, la **norme** n'en porte pas (elle est fixée à 1). Comparer des sens revient donc à comparer des angles.
- Un cosinus ne se lit pas dans l'absolu. Une valeur de 0,6 peut être « très proche » pour un modèle et « banale » pour un autre. C'est pour cela que le pipeline raisonne surtout en **creux relatifs** (§5) et non sur une valeur absolue. La seule valeur absolue qu'il emploie est le seuil de 0,88 du §7.

---

## 5. Détecter les changements de sujet

C'est le cœur du pipeline, dans l'esprit de l'algorithme **TextTiling** (Hearst, 1997) : une frontière thématique apparaît là où le vocabulaire (ici : le sens) change nettement entre la fenêtre qui précède et celle qui suit. L'adaptation consiste à remplacer les fréquences de mots par des plongements.

### 5.1 Blocs et pondération

Le texte est découpé en **blocs** de `BLOCK_SEC` secondes (30 par défaut), chacun vectorisé en $e_k$ (normalisé). On note $w_k$ le nombre de mots du bloc $k$ (au moins 1) : un bloc long compte davantage qu'un bloc de deux mots.

### 5.2 Similarité de part et d'autre d'un point

Soit $W$ la demi-fenêtre en blocs : $W = \max\bigl(2,\ \mathrm{round}(\text{TOPIC\_WINDOW\_SEC}/\text{BLOCK\_SEC})\bigr)$. Avec les valeurs par défaut, $W = 180/30 = 6$ blocs. Pour chaque point de coupure possible $i$ (entre le bloc $i-1$ et le bloc $i$) :

$$
L_i = \sum_{k=i-W}^{i-1} w_k\, e_k, \qquad R_i = \sum_{k=i}^{i+W-1} w_k\, e_k, \qquad s_i = \cos(L_i, R_i) = \frac{L_i \cdot R_i}{\Vert L_i\Vert\,\Vert R_i\Vert}.
$$

Les sommes pondérées sont proportionnelles aux *centroïdes* des deux fenêtres, et le cosinus ignore le facteur d'échelle. $s_i$ répond donc à : « Ce qui précède pendant `TOPIC_WINDOW_SEC` ressemble-t-il à ce qui suit ? ». Près des bords, la fenêtre est tronquée (`max(0, i - window)`, `min(n, i + window)`). Le code est `boundaries.similarity_curve`.

### 5.3 Lissage

La courbe $s_i$ est bruitée. Une **moyenne mobile sur 3 points** la lisse (les bords sont répétés) :

$$
\tilde s_i \;=\; \frac{s_{i-1} + s_i + s_{i+1}}{3}.
$$

Code : `boundaries.smooth_curve`.

### 5.4 Profondeur de creux

Un changement de sujet fait **plonger** $\tilde s$ par rapport à ses voisins. Pour chaque $i$, on « monte » à gauche et à droite tant que la courbe augmente, jusqu'au pic le plus proche, de valeurs $\tilde s_{g(i)}$ et $\tilde s_{d(i)}$ :

$$
d_i \;=\; \bigl(\tilde s_{g(i)} - \tilde s_i\bigr) \;+\; \bigl(\tilde s_{d(i)} - \tilde s_i\bigr).
$$

Exemple minimal : si la courbe vaut $0{,}80$ (pic gauche), descend jusqu'à $0{,}40$ au point $i$, puis remonte à $0{,}78$ (pic droit), alors $d_i = (0{,}80 - 0{,}40) + (0{,}78 - 0{,}40) = 0{,}78$. Un creux peu profond (de 0,80 à 0,75 puis 0,80) donne $0{,}10$.

**Pourquoi la profondeur plutôt que la valeur de $s_i$ ?** Une prédication très cohérente peut avoir une similarité de fond à 0,8 et une autre à 0,5 sans que cela dise quoi que ce soit des changements de sujet. La profondeur mesure le creux **par rapport au voisinage immédiat** et neutralise le niveau de base. Code : `boundaries.depth_scores`.

### 5.5 Choisir les frontières

Dans `boundaries.pick_boundaries`, avec $T$ la durée totale :

1. **Candidats** : les maxima locaux de $d$ avec $d_i > 0$.
2. **Seuil** : on ne retient que $d_i \ge \mu + \lambda\,\sigma$, où $\mu$ et $\sigma$ sont la moyenne et l'écart-type de $d$ et $\lambda = $ `TOPIC_STRICTNESS`. Si $d$ était gaussien, $\lambda = 1$ garderait environ 16 % des points et $\lambda = 2$ environ 2,3 % (**indication seulement** : la distribution réelle est asymétrique, avec beaucoup de petites valeurs et quelques grands creux).
3. **Nombre visé** : $n_{\text{cible}} = \mathrm{round}\bigl(T/\text{TARGET\_SECTION\_SEC}\bigr) - 1$. Sur CPJ 2026 ($T = 12\,454$ s, cible 480 s) : $\mathrm{round}(25{,}95) - 1 = 25$ coupures visées, soit 26 sections.
4. **Sélection gloutonne** : on parcourt les candidats du plus profond au moins profond, et on accepte chacun s'il est à au moins `MIN_SECTION_SEC` du début, de la fin et de toute frontière déjà choisie, jusqu'à atteindre $n_{\text{cible}}$.
5. **Coupures forcées** : tant qu'une section dépasse `MAX_SECTION_SEC`, on la coupe au creux le plus profond situé au moins `MIN_SECTION_SEC` de chaque bord, ou au milieu s'il n'y en a pas.

$n_{\text{cible}}$ est une **borne supérieure**, pas une promesse. Sur CPJ 2026, la contrainte d'écart minimal (au plus $T/300 \approx 41$ candidats) et la fusion des suites (§7) ont abouti à **19 sections** (durée moyenne 10,9 min, de 5,8 à 19,1 min, relevées dans `outputs/cpj2026-day1/sections.json`).

C'est un problème d'optimisation combinatoire que l'on résout par un algorithme **glouton** : simple, rapide, mais sans garantie d'optimum global. Pour l'usage, le choix des creux les plus nets d'abord est un bon compromis.

### 5.6 Caler sur une pause

La frontière trouvée tombe sur un bloc de 30 s, pas sur la fin d'une phrase. `boundaries.snap_to_pause` examine les segments dont le début est à moins de `BLOCK_SEC` secondes du point choisi et maximise

$$
\text{score}(j) \;=\; \underbrace{\max(0,\ \text{début}_j - \text{fin}_{j-1})}_{\text{durée de la pause, en s}} \;+\; \underbrace{0{,}5 \cdot \mathbb{1}[\text{le segment } j-1 \text{ finit par } .\ ?\ !\ …]}_{\text{bonus de fin de phrase}} \;-\; 0{,}01\,\lvert \Delta t \rvert.
$$

Lecture : une pause de 0,5 s vaut autant qu'une fin de phrase ; s'éloigner de 10 s du point choisi coûte 0,1, soit l'équivalent d'une pause de 0,1 s. Les trois coefficients (0,5 ; 0,01 ; le rayon) sont des **choix d'ingénierie**, pas des résultats théoriques : ce sont de bons candidats à tester si les coupes tombent en milieu de phrase.

### 5.7 Complexité

Pour $n$ blocs, $W$ la demi-fenêtre et $d = 1024$ la dimension, le calcul de $s$ coûte $O(n\,W\,d)$ : pour CPJ 2026, $412 \times 12 \times 1024 \approx 5\times10^6$ opérations, négligeable. **Le coût est dans le calcul des plongements** (412 passages dans un transformeur de 24 couches), pas dans les formules ci-dessus.

---

## 6. Le LLM : thèmes et descriptions

### 6.1 Un modèle autorégressif

Qwen2.5 (Qwen Team, 2024) est un transformeur *décodeur seul* d'environ 7,6 milliards de paramètres dans la version utilisée. Comme au §3.1 :

$$
p(y) \;=\; \prod_{t=1}^{T} p\bigl(y_t \mid y_{<t}\bigr),
$$

où ce qui précède $y_{<t}$ contient **le prompt** (consignes, thèmes précédents, transcription de la section) puis les jetons déjà écrits.

### 6.2 Température et décodage

Les scores bruts (*logits*) $z_i$ deviennent des probabilités par un softmax **à température** $T$ :

$$
p_i(T) \;=\; \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

- $T \to 0$ : la distribution se concentre sur le jeton le plus probable (décodage **glouton**, déterministe).
- $T = 1$ : distribution brute du modèle.
- $T > 1$ : distribution aplatie, texte plus varié (et plus risqué).

Dans le code : le backend `transformers` utilise `do_sample=False` (glouton) ; les backends `openai` et `anthropic` envoient `temperature=0.2`, et le `Modelfile` fixe aussi `temperature 0.2`. On veut un résultat **stable et fidèle**, pas créatif, car la sortie doit être un JSON bien formé.

### 6.3 La fenêtre de contexte

Le modèle ne lit qu'un nombre limité de jetons à la fois. L'attention coûte $O(n^2)$ (§3.2) : doubler le texte multiplie le calcul d'attention par environ 4. Le pipeline plafonne donc l'entrée avec `LLM_MAX_CHARS = 20 000` caractères (le début et la fin sont gardés si la section est plus longue). Le commentaire du `Modelfile` estime cela à **environ 6 000 jetons** (soit 3 à 4 caractères par jeton en français, **règle empirique** qui dépend du tokenizer). À cela s'ajoutent le prompt (quelques centaines de jetons) et la réponse (`max_tokens=700`) : le tout reste sous les **12 288** jetons de contexte configurés (`PARAMETER num_ctx 12288`). Avec les 4 096 par défaut d'Ollama, la section serait tronquée **sans erreur**, d'où ce réglage.

**Test de bon sens :** si tu augmentes `LLM_MAX_CHARS` à 40 000, il faut passer `num_ctx` à au moins $2\times 6000 + 1000 \approx 13\,000$, donc 16 384 pour rester sur une puissance de 2.

### 6.4 Quantification : faire tenir 7,6 milliards de nombres en mémoire

Chaque paramètre pèse 2 octets en `float16`. Pour $N = 7{,}6\times10^{9}$ paramètres : $N \times 2 \approx 15$ Go (c'est le « premier téléchargement d'environ 15 Go » du notebook). La **quantification 4 bits** (QLoRA, Dettmers et al., 2023 : type de données *NormalFloat 4*, NF4) stocke chaque poids sur 4 bits, soit un demi-octet :

$$
\text{mémoire} \;\approx\; N \times \frac{4}{8}\ \text{octet} \;\approx\; 3{,}8\ \text{Go},
$$

auxquels s'ajoutent les facteurs d'échelle (un par petit groupe de poids), les activations et le cache d'attention. Le principe : les poids sont répartis en groupes, chaque groupe est mis à l'échelle, puis chaque poids est remplacé par la valeur la plus proche parmi 16 niveaux choisis pour épouser une loi gaussienne (adaptée à la distribution réelle des poids). On perd un peu de précision en échange d'un facteur 4 de mémoire.

Côté Ollama, le même modèle quantifié occupe **4,7 Go** sur disque (valeur lue par `ollama list` sur cette machine ; format `Q4_K_M`, un peu plus gros que du 4 bits pur à cause de blocs gardés à plus haute précision).

---

## 7. Fusion des suites et résumé sans LLM

### 7.1 Garde-fou par similarité de thèmes

Après avoir rédigé le thème de la section $k$, le code vectorise (§4) le thème précédent $\tau_{k-1}$ et le nouveau $\tau_k$, puis calcule $\cos(\tau_{k-1}, \tau_k)$. Si

$$
\cos(\tau_{k-1}, \tau_k) \;\ge\; \text{THEME\_SIM\_CONTINUE} = 0{,}88,
$$

les deux sections sont considérées comme **une suite**, même si le LLM a répondu « non ». Un seuil de 0,88 correspond à un **angle de $\arccos(0{,}88) \approx 28{,}4^\circ$** entre les deux vecteurs.

Abaisser le seuil fusionne davantage (risque : mélanger deux sujets proches) ; l'élever en fusionne moins (risque : une même notion coupée en deux chapitres « partie 1/2 » et « partie 2/2 »). Ce seuil se calibre **sur des exemples** : voir l'exercice 3 du §9.

### 7.2 Résumé extractif (sans LLM)

Sans LLM, `themes.extractive_describe` choisit les phrases les plus *centrales* de la section. Avec $E_j$ le vecteur normalisé de la phrase $j$ et $\bar E$ la moyenne de toutes les phrases :

$$
\text{score}_j \;=\; E_j \cdot \bar E \;=\; \Vert \bar E \Vert \cdot \cos(E_j, \bar E).
$$

Comme $\Vert\bar E\Vert$ est une constante pour toutes les phrases, **classer par $\text{score}_j$ revient à classer par cosinus avec le centroïde**. La phrase de score maximal devient le thème et les quatre meilleures forment la description. C'est un résumé simple (une forme de *centralité*), lisible mais moins fin qu'un LLM.

---

## 8. La vidéo : H.264, images clés, découpe

### 8.1 Les sections sont contiguës

`video.video_bounds` fait commencer la première section à 0 et finir chacune où la suivante commence, la dernière finissant à la durée totale. Les sections forment donc une **partition** de $[0, T]$ : pas de trou, pas de recouvrement.

### 8.2 H.264 et le facteur de débit constant (CRF)

Le mode `reencode` utilise `libx264 -crf 20`. H.264 (Wiegand et al., 2003) compresse en transformant des blocs d'image, puis en **quantifiant** leurs coefficients : on divise par un pas $Q_{\text{step}}$ et on arrondit. Un pas plus grand jette plus de détails et réduit le débit. Le pas dépend du paramètre de quantification $QP$ :

$$
Q_{\text{step}}(QP) \;=\; 2^{(QP-4)/6}.
$$

Le pas **double tous les 6 crans** de $QP$. Le CRF d'x264 fait varier ce $QP$ image par image pour viser une qualité perçue constante, mais l'échelle est la même.

**Mesure sur ce dépôt** (extrait de 20 s de `CPJ2026-day1.mp4`, 640×360, 30 images/s, `-preset veryfast`, sans audio) :

| CRF | Taille           |
| --- | ---------------- |
| 14  | 2 364 218 octets |
| 20  | 1 084 875 octets |
| 26  | 509 239 octets   |
| 32  | 240 085 octets   |

Chaque tranche de **+6 CRF** divise la taille par environ **2,1 à 2,2** (rapports 0,46 à 0,47). La **règle empirique** « +6 CRF ≈ moitié moins de données » se vérifie donc ici. Elle dépend du contenu : une image très détaillée ou très bruitée ne se comporte pas pareil. Pour comparer sur tes propres vidéos, voir l'exercice 5.

### 8.3 Images clés et mode `copy`

Une vidéo compressée n'est pas une suite d'images indépendantes : seules les **images clés** (*I-frames*) sont décodables seules, les autres dépendent de leurs voisines. Avec `-c copy`, ffmpeg ne ré-encode pas ; il ne peut commencer une section qu'**à une image clé**. L'erreur de coupe est donc au plus la distance entre deux images clés. Par défaut, x264 en place une toutes les 250 images (`keyint=250`), soit $250 / 30 \approx 8{,}3$ s à 30 images/s : c'est l'origine de « précis à quelques secondes près » dans le commentaire de `CUT_MODE`. Le mode `reencode` recrée une image clé exactement au point voulu, au prix du temps de calcul.

---

## 9. S'en imprégner et ajuster

### 9.1 Tableau : réglage → formule → effet → que faire

| Réglage               | Ce qu'il change dans les formules                    | Effet attendu si on l'augmente                                         | Symptôme → réglage                                                                   |
| ---------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `BLOCK_SEC`          | taille des blocs (§5.1), nombre de points candidats | courbe plus lisse, frontières plus grossières                        | frontières imprécises → baisser (20 s) ; très bruité → monter (45 s)              |
| `TOPIC_WINDOW_SEC`   | demi-fenêtre $W$ (§5.2)                          | $s_i$ moyenne plus de texte : moins sensible aux digressions courtes | trop de coupures sur des digressions → monter ; sujets courts ratés → baisser        |
| `TOPIC_STRICTNESS`   | seuil $\mu + \lambda\sigma$ (§5.5)                | ne garde que les creux les plus nets                                   | sections trop nombreuses → monter (0,5 puis 1)                                         |
| `MIN_SECTION_SEC`    | écart minimal entre frontières                     | pas de section plus courte                                             | sections trop courtes → monter                                                         |
| `TARGET_SECTION_SEC` | $n_{\text{cible}}$ (§5.5)                         | moins de coupures visées                                              | trop de sections → monter (600)                                                        |
| `MAX_SECTION_SEC`    | déclenche les coupures forcées, limite la fusion   | autorise de longues sections                                           | notion longue coupée → monter                                                         |
| `THEME_SIM_CONTINUE` | seuil cosinus (§7.1)                                | fusionne moins                                                         | suites non fusionnées → baisser (0,85) ; sujets distincts fusionnés → monter (0,92) |
| `LLM_MAX_CHARS`      | longueur du contexte (§6.3)                         | le LLM lit plus de texte, plus de calcul                               | thèmes hors sujet sur de longues sections → monter, avec `num_ctx`                  |
| `CHAPTER_TITLE_MAX`  | troncature des titres                                | titres plus longs                                                      | titres coupés → monter                                                                |
| `WHISPER_MODEL`      | taille du transformeur (§3.6)                       | WER plus bas, calcul plus long                                         | noms propres mal transcrits → `medium`, plus `INITIAL_PROMPT`                      |
| `CUT_MODE`           | `copy` ou ré-encodage (§8.3)                     | `reencode` : coupes exactes                                          | coupes décalées → `reencode`                                                       |

### 9.2 Une méthode pour ajuster sans se perdre

1. **Un seul réglage à la fois.** Si tu en changes deux, tu ne sais plus lequel a produit l'effet.
2. **Réutilise le cache.** Avec `FORCE = {"themes"}`, la transcription (l'étape la plus longue) n'est pas refaite : tu peux tester les frontières en quelques minutes. Si tu veux seulement les frontières, mets `LLM_BACKEND = "none"`.
3. **Regarde des nombres, pas seulement le plan.** Relève le nombre de sections et la distribution des durées (moyenne, minimum, maximum), comme au §5.5. Si la moyenne dépasse nettement `TARGET_SECTION_SEC`, c'est que la contrainte d'écart minimal ou les fusions dominent.
4. **Fais-toi une vérité terrain.** Note à la main, sur une vidéo, les instants où le sujet change vraiment. Compare ensuite avec les frontières du pipeline, avec une tolérance (par exemple ±30 s) : une frontière est un *succès* si elle tombe à moins de la tolérance d'une frontière de référence. Tu obtiens alors

   $$
   P = \frac{\text{frontières correctes}}{\text{frontières prédites}}, \qquad R = \frac{\text{frontières de référence retrouvées}}{\text{frontières de référence}}, \qquad F_1 = \frac{2PR}{P+R}.
   $$

   $P$ basse : trop de coupures (monter `TOPIC_STRICTNESS`). $R$ basse : des changements de sujet ratés (baisser `TOPIC_STRICTNESS` ou `TARGET_SECTION_SEC`). C'est la seule façon de dire « ce réglage est meilleur » sans se fier à l'impression.
5. **Garde un journal** : réglage, nombre de sections, $F_1$, remarque. Trois lignes par essai suffisent.

### 9.3 Parcours d'étude

Dans l'ordre, en s'arrêtant quand tu as ce qu'il te faut :

1. **Algèbre linéaire minimale** : vecteur, produit scalaire, norme, angle. Suffit pour les §4, §5 et §7. *Référence :* Deisenroth, Faisal et Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020), chapitres 2 et 3, disponible gratuitement en ligne.
2. **Probabilités et softmax** : produit de probabilités conditionnelles, logarithme, température. Suffit pour les §3 et §6. *Référence :* Jurafsky et Martin, *Speech and Language Processing* (3e éd., version préliminaire en ligne), chapitres sur les modèles de langage et la reconnaissance de la parole.
3. **Attention et transformeur** : le mécanisme du §3.2, puis l'architecture complète. *Référence :* Vaswani et al., « Attention Is All You Need » (2017).
4. **Chaque brique dans son article d'origine**, dans l'ordre où elle apparaît dans le pipeline :
   - Radford et al., « Robust Speech Recognition via Large-Scale Weak Supervision » (Whisper, 2022) ;
   - Hearst, « TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages », *Computational Linguistics* 23(1), 1997 ;
   - Reimers et Gurevych, « Sentence-BERT » (2019), pour la façon d'obtenir un vecteur par phrase ;
   - Chen et al., « BGE M3-Embedding » (2024) ;
   - Qwen Team, « Qwen2.5 Technical Report » (2024) ;
   - Dettmers et al., « QLoRA: Efficient Finetuning of Quantized LLMs » (2023), pour NF4 ;
   - Wiegand et al., « Overview of the H.264/AVC Video Coding Standard », *IEEE Trans. Circuits Syst. Video Technol.*, 2003.

### 9.4 Exercices sur les données du dépôt

Les exercices 1 à 3 utilisent `outputs/cpj2026-day1/` (transcription et sections déjà calculées). Lance-les depuis la racine du dépôt.

> **Mémoire et temps.** Calculer les plongements des 412 blocs avec `bge-m3` prend **plusieurs minutes sur CPU** et quelques Go de RAM. Ferme les autres applications lourdes. Les plongements sont enregistrés dans `emb.npy` : les essais suivants sont instantanés.

#### Exercice 1 — Voir la courbe de similarité et les creux

```python
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from pipeline.config import Config
from pipeline import boundaries as B

cfg = Config()                          # puis essaie : Config(TOPIC_WINDOW_SEC=120, TOPIC_STRICTNESS=0.5)
env = SimpleNamespace(device="cpu")
units = json.load(open("outputs/cpj2026-day1/transcript.json"))["units"]
blocks = B.make_blocks(units, cfg.BLOCK_SEC)

if Path("emb.npy").exists():
    emb = np.load("emb.npy")
else:
    emb = B.embed_texts([b["text"] for b in blocks], cfg, env, show=True)
    np.save("emb.npy", emb)

weights = np.array([max(1, len(b["text"].split())) for b in blocks], float)
btimes = np.array([b["start"] for b in blocks])
window = max(2, round(cfg.TOPIC_WINDOW_SEC / cfg.BLOCK_SEC))

sims = B.similarity_curve(emb, weights, window)          # s_i      (§5.2)
smooth = B.smooth_curve(sims)                            # s~_i     (§5.3)
depth = B.depth_scores(emb, weights, window)             # d_i      (§5.4)
picked = B.pick_boundaries(depth, btimes, units[0]["start"], units[-1]["end"], cfg.MIN_SECTION_SEC,
                           cfg.TARGET_SECTION_SEC, cfg.MAX_SECTION_SEC, cfg.TOPIC_STRICTNESS)   # §5.5

print(f"{len(blocks)} blocs, fenêtre W = {window}, {len(picked)} frontières retenues")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
a1.plot(btimes / 60, sims, alpha=0.4, label="s_i")
a1.plot(btimes / 60, smooth, label="lissée")
a1.legend()
a2.plot(btimes / 60, depth, color="tab:red", label="profondeur d_i")
a2.axhline(depth[1:].mean() + cfg.TOPIC_STRICTNESS * depth[1:].std(), ls="--", color="gray", label="seuil μ + λσ")
for i in picked:
    a2.axvline(btimes[i] / 60, color="tab:green", alpha=0.5)
a2.set_xlabel("minutes")
a2.legend()
plt.tight_layout()
plt.savefig("courbes_similarite.png", dpi=120)
```

*À observer :* les traits verts tombent-ils sur les pics rouges ? Les pics les plus hauts correspondent-ils à de vrais changements de sujet quand tu lis la transcription autour de ce moment ?

#### Exercice 2 — Faire varier les réglages et compter

Réutilise les variables de l'exercice 1 (sans recalculer les plongements) :

```python
t0, t1 = units[0]["start"], units[-1]["end"]
for window_sec in (90, 180, 300):
    for lam in (0.0, 0.5, 1.0):
        w = max(2, round(window_sec / cfg.BLOCK_SEC))
        d = B.depth_scores(emb, weights, w)
        p = B.pick_boundaries(d, btimes, t0, t1, cfg.MIN_SECTION_SEC, cfg.TARGET_SECTION_SEC, cfg.MAX_SECTION_SEC, lam)
        cuts = [t0] + [btimes[i] for i in p] + [t1]
        dur = np.diff(cuts) / 60
        print(f"fenêtre {window_sec:>3}s  λ={lam:.1f} : {len(p) + 1:>2} sections, durée moyenne {dur.mean():4.1f} min "
              f"(min {dur.min():4.1f}, max {dur.max():4.1f})")
```

*À observer :* comment $\lambda$ réduit-il le nombre de sections ? Pour quelle fenêtre les sections deviennent-elles trop longues et déclenchent-elles les coupures forcées ?

#### Exercice 3 — Calibrer le seuil 0,88 sur tes propres thèmes

Cet exercice est léger (une vingtaine de phrases à vectoriser) :

```python
import json
import numpy as np
from types import SimpleNamespace
from pipeline.config import Config
from pipeline import boundaries as B

cfg, env = Config(), SimpleNamespace(device="cpu")
themes = [s["theme"] for s in json.load(open("outputs/cpj2026-day1/sections.json"))["sections"]]
E = B.embed_texts(themes, cfg, env)
cos = [float(E[i] @ E[i + 1]) for i in range(len(E) - 1)]       # thèmes consécutifs
for i, c in enumerate(cos, 1):
    print(f"{i:>2} → {i + 1:>2} : cos = {c:.2f}   {'(>= 0.88 : suite)' if c >= cfg.THEME_SIM_CONTINUE else ''}")
print("médiane :", round(float(np.median(cos)), 2), "| max :", round(max(cos), 2))
```

*À observer :* parmi les paires, lesquelles sont vraiment des suites (lis les deux thèmes) ? Quelle valeur de seuil les séparerait des autres paires ?

#### Exercice 4 — Mesurer le WER

Transcris deux minutes d'une vidéo avec `small` puis `medium`. Corrige une des transcriptions à la main pour obtenir la référence, puis calcule la distance d'édition au niveau des mots (le paquet `jiwer` fait `jiwer.wer(reference, hypothese)`). Compare le gain de $\mathrm{WER}$ et le temps de calcul : le gain justifie-t-il le facteur de lenteur ?

#### Exercice 5 — Vérifier la règle « +6 CRF ≈ moitié »

```bash
for c in 14 20 26 32; do
  ffmpeg -y -hide_banner -loglevel error -ss 600 -i data/MaVideo.mp4 -t 20 -an \
         -c:v libx264 -preset veryfast -crf $c -pix_fmt yuv420p /tmp/crf_$c.mp4
  echo "crf $c : $(stat -c %s /tmp/crf_$c.mp4) octets"
done
```

Calcule les rapports de taille entre CRF successifs et compare avec 0,46 à 0,47 mesuré au §8.2. Essaie un passage très détaillé et un passage presque fixe (un orateur sur fond uni) : le rapport reste-t-il proche de 0,5 ?

---

## Références

- Nyquist, H. (1928) et Shannon, C. E. (1949), travaux fondateurs sur l'échantillonnage.
- Sakoe, H. et Chiba, S. (1978). *Dynamic programming algorithm optimization for spoken word recognition*. IEEE Trans. ASSP.
- Hearst, M. A. (1997). *TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages*. Computational Linguistics 23(1).
- Wiegand, T. et al. (2003). *Overview of the H.264/AVC Video Coding Standard*. IEEE TCSVT.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Reimers, N. et Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.
- Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216.
- Qwen Team (2024). *Qwen2.5 Technical Report*. arXiv:2412.15115.
- Deisenroth, M. P., Faisal, A. A. et Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press.
- Jurafsky, D. et Martin, J. H. *Speech and Language Processing* (3e éd., en ligne).
