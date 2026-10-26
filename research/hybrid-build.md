# Build hybride Cython + Rust pour scikit-allel2

Ticket : [#4](https://github.com/ltalignani/scikit-allel2/issues/4), carte [#1](https://github.com/ltalignani/scikit-allel2/issues/1).
Date des constats : 2026-10-26. Versions relevées sur PyPI et dans les dépôts à cette date.

## Réponse courte

1. **Backend de build : setuptools, enrichi par `setuptools-rust`.** Cython reste compilé par `ext_modules` (comme aujourd'hui), le module PyO3 est déclaré via le mot-clé `rust_extensions`. Les deux coexistent dans un même `setup()`.
2. **maturin est écarté pour le package hybride.** maturin ne compile que du Rust (pyo3, cffi, uniffi) : aucune mention de Cython dans sa documentation ni dans son code. Il reste pertinent si l'on sépare un jour le noyau Rust dans un package dédié.
3. **abi3 : non pour la première itération.** NumPy n'est pas encore compatible limited API de bout en bout (chantier en cours côté NumPy). On produit des wheels par version de Python (cp311 à cp314, plus cp314t). abi3 peut faire l'objet d'un spike ultérieur.
4. **rust-numpy** supporte NumPy 2 depuis la 0.22. Le support free-threaded (3.13t) date de la 0.24. Le MSRV est Rust 1.85.
5. **cibuildwheel** : Rust n'est pas installé dans les images manylinux, il faut l'installer dans `CIBW_BEFORE_ALL_LINUX`. Le décalage de cible macOS (Rust 10.12 contre CPython 10.9 avant 3.12) est à traiter.
6. **Point bloquant indirect : NumPy 2.5.x déclare `requires_python >=3.12`.** Le plancher Python 3.11 de la carte impose de plafonner NumPy à la série 2.4 (2.4.4 publie encore des wheels cp311 et cp314t).

## Constats détaillés

### 1. setuptools-rust contre maturin

**maturin** (https://github.com/PyO3/maturin) :
- Build de crates Rust avec les bindings pyo3, cffi et uniffi. La documentation parle de « mixed rust/python project » (le dossier Python du module est à côté de `Cargo.toml`) et de `python-source` pour éviter un piège d'`ImportError`.
- Aucune mention de Cython, d'extensions setuptools `ext_modules`, d'abi3 ou de free-threaded dans le README.
- Recherche de code `cython` dans le dépôt `PyO3/maturin` : aucun résultat. maturin n'a donc aucun mécanisme pour compiler des `.pyx`.
- Conséquence : pour `allel2`, maturin obligerait à sortir Cython du build (réécriture ou second système), ce qui va à l'encontre de la décision « Cython et Rust coexistent » de la carte.

**setuptools-rust** (https://github.com/PyO3/setuptools-rust) :
- Plugin setuptools. Le mot-clé `rust_extensions` est enregistré comme point d'entrée setuptools (`setuptools_rust/setuptools_ext.py`, fonction `rust_extensions`). Exemple officiel : `setup.py` avec `rust_extensions=[...]` (dossier `examples/hello-world-setuppy`).
- Le README documente la configuration en `pyproject.toml` (`[[tool.setuptools-rust.ext-modules]]`, avec les paramètres de `RustExtension` en tirets). Le README ne montre pas l'intégration avec `ext_modules`. Le code source, lui, montre une commande `build_rust` qui hérite des réglages de `build_ext` (`setuptools_rust/build.py`, lignes 107 à 115), et calcule le nom du `.so` via `build_ext.get_ext_fullpath`. Les deux commandes se partagent donc le même build.
- Ajout à l'existant : `build-system.requires` gagne `setuptools-rust` (version actuelle 1.13.0, `requires_python >=3.9`). Le `setup.py` garde `ext_modules` pour Cython et ajoute `rust_extensions`.
- Piège à connaître : `py_limited_api` de `RustExtension` vaut `auto` par défaut et suit alors `bdist_wheel.py_limited_api` (`build.py`, lignes 491 et 525 à 532). Si abi3 était activé pour les wheels, la partie Rust passerait aussi en abi3 sans l'avoir demandé. Le changelog indique que le paramètre explicite `py_limited_api` est déprécié depuis 1.9.0 au profit de cette configuration automatique.

Sources : README et `setuptools_rust/setuptools_ext.py`, `setuptools_rust/build.py` (dépôt PyO3/setuptools-rust) ; CHANGELOG brut https://raw.githubusercontent.com/PyO3/setuptools-rust/main/CHANGELOG.md (1.9.0 déprécie `py_limited_api`, 1.10.0 supporte 3.13, 1.11.0 supporte free-threaded). Version PyPI actuelle : 1.13.0 (https://pypi.org/project/setuptools-rust/).

### 2. Cython et le module Rust dans un même wheel

- Le projet a aujourd'hui trois fichiers `.pyx` qui font `cimport numpy` (`allel2/opt/io_vcf_read.pyx`, `model.pyx`, `stats.pyx`). Ce sont eux qui limitent l'abi3, pas Rust.
- Cython documente le Limited API : activation par la macro `Py_LIMITED_API` (valeur = version minimale ciblée, par exemple `0x030B0000` pour 3.11). Des fonctionnalités ne sont pas supportées : profilage, line tracing, `cdef class` héritant de types natifs, accès direct à `array.array`. Les memoryviews typées demandent Python 3.11+ en mode limited API.
  Source : https://cython.readthedocs.io/en/latest/src/userguide/limited_api.html
- Cython 3.1.0 : support du free-threaded (3.13t) ; « il suffit de définir `Py_LIMITED_API` » pour activer le limited API. Cython 3.3.0 : adaptations pour Python 3.15.
  Source : https://github.com/cython/cython/blob/master/CHANGES.rst (dernière version datée : 3.3.0, 2026-08-22).
- Limite amont : le support Py_LIMITED_API + free-threaded (PEP 803, Python 3.15) est encore un chantier ouvert côté Cython : https://github.com/cython/cython/issues/7399.
- Compilation mixte : Cython et PyO3 produisent chacun un `.so` distinct. Aucun conflit de symboles à attendre, les deux modules sont indépendants (ils ne partagent qu'un Python). Non testé sur allel2.

### 3. abi3 : faisabilité

**Côté PyO3 et rust-numpy (partie Rust)** :
- PyO3 accepte `abi3-pyXY` (compilé comme si pour la version XY, utilisable sur les versions plus récentes) et, depuis 0.29, `abi3t` (PEP 803, Python 3.15t+). Source : https://pyo3.rs/main/building-and-distribution/multiple-python-versions et CHANGELOG PyO3 (https://github.com/PyO3/pyo3/blob/main/CHANGELOG.md : 0.28.0 « Support for free-threaded Python is now opt-out », 0.29.0 « Drop support for Python 3.13t », 0.29.0 `abi3t`, dernière version 0.30.0 datée du 2026-10-13).
- rust-numpy : 0.29.0 « The NumPy C API binding has been updated to target the ABI v2 » ; 0.30.0 « Add support for free-threaded stable abi (abi3t) from Python 3.15t+ ». Source : https://github.com/PyO3/rust-numpy/blob/main/CHANGELOG.md (brut : https://raw.githubusercontent.com/PyO3/rust-numpy/main/CHANGELOG.md). Le README ne mentionne pas abi3 ni free-threaded.

**Côté NumPy (partie C)** :
- NumPy ne publie pas de wheels abi3. Son repo a un tracking issue « Adoption of CPython limited API » (https://github.com/numpy/numpy/issues/31913, ouvert au 2026-10-03) : l'objectif est de migrer module par module vers `Py_LIMITED_API`. Le module `_multiarray_umath` n'est pas porté (bloqué par les heap types). Une option de build `python.allow_limited_api` existe (PR #31910, fusionnée).
- Issue #30704 (fermée le 2026-05-16) : « NumPy doesn't use the Limited C API and Stable ABI itself, but does support other projects targeting the Stable ABI while using the NumPy headers. Support for that was implemented about two years ago in gh-25531, shipped in NumPy 2.0. It isn't widely used yet, but seems to work fine. » Source : https://github.com/numpy/numpy/issues/30704 et https://github.com/numpy/numpy/pull/25531 (fusionnée le 2024-01-16).
- Conclusion : une extension tierce peut en théorie cibler abi3 avec les headers NumPy 2.x. Côté Cython, la combinaison `cimport numpy` + `Py_LIMITED_API` n'a pas été testée ici, et le PR gh-25531 indique lui-même que Cython avait encore des lacunes à l'époque. Il faudrait un spike pour valider.

**Verdict abi3** : écarter pour la première itération. Coût : on construit une wheel par version (cp311, cp312, cp313, cp314) et une cp314t, soit 5 wheels par plateforme. Gain potentiel de abi3 : une seule wheel pour 3.11 à 3.14 (et il faudrait aussi une wheel free-threaded séparée, car abi3 classique ne couvre pas les builds free-threaded : PEP 803 et abi3t sont nécessaires pour cela, et PyO3 ne propose abi3t qu'à partir de 3.15t).

### 4. rust-numpy avec NumPy 2 et free-threading

- NumPy 2 : support depuis rust-numpy 0.22 (« Add support for NumPy v2 which had a number of changes to the C API »).
- Free-threaded : 3.13t depuis 0.24 (« Support Python 3.13t free-threaded Python »), 0.23 exige `Element: Sync`.
- MSRV : Rust 1.85 (rust-numpy 0.30, « raise MSRV to 1.85 (following PyO3) »).
- Python : le README de PyO3 indique « PyO3 supports all actively-supported Python 3 and PyPy versions ». PyO3 0.29 retire 3.13t, mais garde 3.14t.
- Versions à pinner : PyO3 0.30.0 (2026-10-13), rust-numpy 0.30.x (aligné sur PyO3 0.30).

### 5. NumPy et Python 3.11

Mesuré sur PyPI (https://pypi.org/pypi/numpy/json, 2026-10-26) :
- NumPy 2.5.4 (dernière) : `requires_python >=3.12`. Les wheels disponibles : cp312, cp313, cp314, cp314t, cp315, cp315t.
- NumPy 2.4.4 : wheels cp311, cp312, cp313, cp313t, cp314, cp314t, pp311.
- Pour garder Python 3.11 dans la matrice, il faut plafonner `numpy<2.5` (ou `numpy>=2.0,<2.5` pour 3.11). Cela se combine avec la contrainte zarr 3.2 et Python 3.12 déjà relevée dans la carte.

### 6. cibuildwheel

Sources : https://cibuildwheel.pypa.io/en/stable/faq/ et https://cibuildwheel.pypa.io/en/stable/options/.

- **Rust dans manylinux** : « If you build Rust wheels, you need to download the Rust compilers in manylinux. » Exemple : `CIBW_BEFORE_ALL_LINUX: curl -sSf https://sh.rustup.rs | sh -s -- -y`, puis `CIBW_ENVIRONMENT_LINUX: "PATH=$HOME/.cargo/bin:$PATH"`. Sur Windows, ajouter la cible `i686-pc-windows-msvc` si 32 bits.
- **macOS** : « Rust's minimum macOS target is 10.12, while CPython supports 10.9 before Python 3.12 ». Il faut donc fixer le `MACOSX_DEPLOYMENT_TARGET` en conséquence, en vérifiant ce que le build retient.
- **musllinux 32 bits** : Rust ne fournit pas Cargo pour cette cible, à exclure avec `skip`.
- **Plateformes** : macOS `x86_64`, `arm64`, `universal2`. Pour tester une seule architecture d'un universal2, suffixe `:arch`.
- **abi3** : cibuildwheel exécute `abi3audit` sur les wheels abi3 après la réparation (commande par défaut `abi3audit --strict --report {abi3_wheel}`). Sans wheels abi3, rien à faire.
- **Free-threaded** : les wheels sont tagguées `cp314t-...`. La doc de l'option `enable` précise que `cpython-freethreading` n'est pas une valeur valable ; on sélectionne ou exclut les builds free-threaded via `build` et `skip` (exemple : `skip = "cp3??t-*"`). Le défaut exact (inclus ou non) n'est pas explicite dans la doc lue : à vérifier au premier essai.
- **Cython** : la doc cibuildwheel donne l'exemple `"Cython"` dans `build-system.requires` du `pyproject.toml`.
- **Équivalent maturin** : cibuildwheel mentionne `maturin-action` pour les projets Rust, qui « builds the non-Python Rust modules once ». Cette option ne s'applique pas à un setup hybride setuptools.

## Recommandation pour allel2

- `pyproject.toml` : `build-system.requires` ajoute `setuptools-rust` (et garde `cython`, `numpy`). Passer `requires-python` à `>=3.11` (aujourd'hui `>=3.10`), `numpy>=2.0,<2.5` tant que 3.11 est visé.
- `setup.py` : garder la boucle `ext_modules` pour les `.pyx`, ajouter `rust_extensions=[RustExtension("allel2.<nom>", path="<crate>/Cargo.toml", binding=Binding.PyO3)]`. Vérifier avec `pip install -e .` puis `pytest -v allel2`.
- Wheels : une par version (cp311, cp312, cp313, cp314) et cp314t, Linux et macOS (`x86_64`, `arm64`). `CIBW_BEFORE_ALL_LINUX` avec rustup, `skip` pour musllinux 32 bits.
- Pas d'abi3 pour le moment. Si l'on veut l'essayer plus tard : un spike qui compile `io_vcf_read.pyx` avec `Py_LIMITED_API` et vérifie `import` sous 3.11 et 3.14, en gardant en tête le piège du `py_limited_api` automatique de setuptools-rust.
- Prochain ticket utile (non fait ici) : un prototype minimal `setup.py` + un module PyO3 qui compile sur Linux et macOS, puis une exécution cibuildwheel. Aucun build n'a été lancé pour cette note.

## Limites de cette recherche

- Aucun build ni test n'a été exécuté. Les affirmations sur l'intégration setuptools-rust + Cython sont tirées du code source, pas d'une démonstration.
- Le support abi3 pour Cython + NumPy n'est pas vérifié (pas de spike).
- La documentation de setuptools-rust (site readthedocs) n'a pas été lue dans sa version actuelle : une page renvoyait 404 au moment de la recherche. Le README et le code source ont servi de référence.
- Le détail exact du défaut des builds free-threaded dans cibuildwheel n'a pas été tranché par la doc.
