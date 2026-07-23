# Auto Rotoscope VSE

Extension Blender de **rotoscopie automatique** pour le **Video Sequence Editor (VSE)**,
basée sur **SAM 2.1** exécuté via **ONNX Runtime** (sans PyTorch).

On clique sur un objet dans l'aperçu, SAM 2.1 génère le masque, la sélection se
**propage** sur la séquence, et on **ajoute/retire des points** pour corriger. La sortie
est une **séquence de mattes alpha** (PNG) réutilisable dans le VSE / le compositing.

- **CPU par défaut**, **GPU exploité si présent** (best-effort).
- **100 % embarqué** : aucune connexion réseau, conforme aux règles de
  `extensions.blender.org` (limite ~200 Mo/ZIP, pas de téléchargement externe).
- Modèle **tiny** embarqué (le seul qui tient sous la limite de 200 Mo une fois
  onnxruntime + OpenCV inclus). Le sélecteur de modèle reste extensible.

> ⚠️ SAM 2 n'a pas d'export ONNX du **mode vidéo** (memory attention — voir
> [facebookresearch/sam2#702](https://github.com/facebookresearch/sam2/issues/702)).
> Le suivi temporel est donc assuré par **flux optique (OpenCV)** : le masque et les
> points de la frame N sont advectés vers N+1, puis SAM ré-affine à chaque frame.

## Périmètre

- **Windows x64** et **Linux x64**, Blender **5.1** minimum (Python 3.13 ; wheels cp313).
  Pour cibler Blender 4.2–5.0 (Python 3.11), régénérer en cp311 — voir `PY_TAG` dans
  `scripts/fetch_assets.py` et `blender_version_min` dans le manifest.
- Uniquement le **VSE** (strips vidéo / séquences d'images).
- **GPU** : DirectML sur Windows (NVIDIA/AMD/Intel, sans CUDA). Linux tourne en CPU
  (le paquet onnxruntime-gpu/CUDA dépasserait la limite de 200 Mo).

## Structure

```
auto_rotoscope/            # le package de l'extension (ce qui est zippé)
  blender_manifest.toml
  __init__.py              # register()/unregister(), keymap
  preferences.py           # device (Auto/CPU/GPU), dossier de sortie
  properties.py            # état de session (points +/-, plage, modèle)
  engine/
    ort_session.py         # sélection de l'ExecutionProvider (DirectML→CPU)
    loader.py              # chargement/cache des sessions ONNX
    sam2.py                # pipeline image SAM 2.1 (encoder/decoder ONNX)
    propagate.py           # propagation par flux optique (Farneback)
  ops/
    common.py              # lecture des frames du strip, aperçu image
    op_pick.py             # picking modal (clic = +, Ctrl+clic = −)
    op_track.py            # propagation + export de la séquence de mattes
    op_clear.py            # reset des points
  ui/
    panel_vse.py           # panneau latéral "SAM2 Roto" dans le VSE
  models/                  # <- ONNX SAM 2.1 tiny (récupéré par le script)
  wheels/                  # <- wheels onnxruntime + opencv (récupérées)
  licenses/                # NOTICE + licences tierces
scripts/
  fetch_assets.py          # télécharge modèles + wheels, imprime le bloc wheels[]
```

## Préparer les dépendances (avant build)

Les gros binaires (wheels + modèle ONNX) ne sont **pas** versionnés. Les récupérer :

```bash
python scripts/fetch_assets.py
```

Le script télécharge le modèle SAM 2.1 tiny (encoder/decoder ONNX) et les wheels
(`onnxruntime-directml` pour Windows, `onnxruntime` pour Linux, `opencv-python-headless`
pour les deux — `numpy` est fourni par Blender), puis imprime la liste `wheels = [...]`
exacte à coller dans `auto_rotoscope/blender_manifest.toml`.

## Build de l'extension

```bash
cd auto_rotoscope
blender --command extension validate
blender --command extension build --split-platforms
# -> un .zip par plateforme (chacun doit rester < 200 Mo)
```

## Installer / tester

`Edit > Preferences > Get Extensions > (menu) Install from Disk…` puis choisir le ZIP.

Dans un espace **Video Editing** : sélectionner un strip vidéo, ouvrir le panneau
latéral (`N`) → onglet **SAM2 Roto**. Étapes : **Pick** (cliquer l'objet, `L` dans
l'éditeur d'image), puis **Track & Export Matte**.

## Notes de conformité (plateforme)

- Aucune permission `network` (tout est embarqué, fonctionne hors-ligne).
- Permission `files` déclarée (écriture des séquences de mattes générées).
- Code lisible, pas de bytecode.
- Wheels non modifiées, issues de PyPI, sous `./wheels/`.
- Licence GPL-3.0-or-later ; modèle SAM 2.1 sous Apache-2.0 (compatible, redistribuable).

## Caveat wheels `abi3`

Les wheels d'OpenCV sont taguées `cp37-abi3`. Certaines versions de Blender ont eu du mal
à reconnaître les tags `abi3` (elles attendaient `cp311`). Si l'extension refuse la wheel
OpenCV, renommer le fichier `...-cp37-abi3-...` en `...-cp311-abi3-...` et ajuster le
manifest en conséquence.

## Licences tierces

- **SAM 2.1** — Apache-2.0 (Meta Platforms, Inc.)
- **onnxruntime / onnxruntime-directml** — MIT (Microsoft)
- **opencv-python-headless** — Apache-2.0

Voir `auto_rotoscope/licenses/NOTICE.txt`.
