# Dtypes chaînes sous NumPy 2, zarr 3 et pandas 3 pour les champs VCF

Ticket : [Dtypes chaînes sous NumPy 2, zarr 3 et pandas 3 : quelles options stables pour les champs VCF ?](https://github.com/ltalignani/scikit-allel2/issues/3), carte [#1](https://github.com/ltalignani/scikit-allel2/issues/1).
Date : 2026-10-26. Environnement mesuré : Python 3.12.6, numpy 2.5.4, zarr 3.4.1, pandas 3.1.0 (sans pyarrow).
Scripts reproductibles : [string-dtypes-exp1.py](string-dtypes-exp1.py) (aller-retour zarr), [string-dtypes-exp2.py](string-dtypes-exp2.py) (allel2, pandas, perfs).

## Réponse courte

- **Écriture** : `vcf_to_zarr` écrit déjà correctement ses chaînes. `dtype='str'` → type zarr v3 **`string`** (codec `vlen-utf8`), spécifié dans zarr-extensions, donc **stable et lisible par d'autres implémentations**. En format v2, il devient `|O` + `vlen-utf8`, identique à ce qu'écrivait scikit-allel avec zarr 2.
- **Lecture** : zarr 3 renvoie **toujours `numpy.dtypes.StringDType`** pour ce type, sans option pour obtenir `object` (code source `zarr/core/dtype/npy/string.py`, classe `VariableLengthUTF8`). C'est vrai aussi pour les anciens zarr v2 `|O`/`vlen-utf8` écrits par scikit-allel. Le choix n'appartient donc pas à `allel2` : tout lecteur zarr 3 obtient `StringDType`.
- **pandas 3** : un tableau `object` de chaînes devient une colonne de dtype **`str`** (`StringDtype(storage='python')` sans pyarrow). Un tableau `StringDType` devient, à l'inverse, une colonne **`object`**.
- **Les 9 échecs de `test_vcf_read` sont des assertions de dtype**, pas des bugs d'écriture : 6 tests `vcf_to_zarr*` comparent `read_vcf` (`object`) à la relecture zarr (`StringDType`), et 3 tests `vcf_to_dataframe*` attendent `object` là où pandas 3 donne `str`. **`vcf_to_zarr` ne dépend pas du backend `chunked`** : il appelle directement `zarr.open_group` / `create_array` (`allel2/io/vcf_read.py`, `vcf_to_zarr` et `_zarr_setup_datasets`).
- **Octets de largeur fixe (`S`)** : en v3, zarr les écrit en `null_terminated_bytes`, **absent de zarr-extensions** ; zarr émet `UnstableSpecificationWarning` (la représentation peut changer, illisible par d'autres bibliothèques). En v2, `|S10` reste stable.

## Faits et sources

### NumPy 2

- `np.dtypes.StringDType` (NEP 55) existe depuis NumPy 2.0 : chaînes UTF-8 de longueur variable stockées hors du tableau, sans objets Python. Disponible sur tout le support visé (Python ≥ 3.11 et NumPy 2).
- `np.array([...str])` sans dtype donne toujours `<U` (largeur fixe UTF-32), pas `StringDType` (mesuré).
- Mesures sur 10⁶ valeurs CHROM (5 modalités) :

| dtype | mémoire | `np.unique` | `a == x` |
|---|---|---|---|
| `object` | 8 Mo + objets Python | 402 ms | 13,9 ms |
| `StringDType` | 16 Mo | 16 ms | 11,0 ms |
| `S2` | 2 Mo | 6,6 ms | 5,3 ms |
| `U2` | 8 Mo | 7,9 ms | 5,5 ms |

  `object` est de loin le plus lent sur le tri et `unique`. `StringDType` est ~25× plus rapide que `object` sur `unique`.
- `allel2.ChromPosIndex` (tri, `locate_range`) fonctionne à l'identique avec `object`, `StringDType` et `S` (mesuré).

### zarr 3 (zarr-python 3.4.1)

Source : `zarr/core/dtype/npy/string.py`, `zarr/core/dtype/npy/bytes.py`, `zarr/core/dtype/common.py` ; liste des types v3 spécifiés : [zarr-extensions/data-types](https://github.com/zarr-developers/zarr-extensions/tree/main/data-types).

Aller-retour mesuré (`create_array` puis `open_array(...)[:]`) :

| dtype écrit | v3 : métadonnée | v3 : relu | v2 : métadonnée | v2 : relu |
|---|---|---|---|---|
| `'str'` (code actuel de `vcf_to_zarr`) | `string` | `StringDType` | `\|O` + vlen-utf8 | `StringDType` |
| `StringDType` | `string` | `StringDType` | `\|O` + vlen-utf8 | `StringDType` |
| `object` | **erreur** (« Zarr data type resolution from object failed ») | | **erreur** | |
| `S10` | `null_terminated_bytes` (**non spécifié**, warning) | `S10` | `\|S10` | `S10` |
| `U8` | `fixed_length_utf32` (spécifié) | `<U8` | `<U8` | `<U8` |

- Types v3 spécifiés dans zarr-extensions : `string` (UTF-8 variable), `bytes` (octets variables), `fixed_length_utf32`. **`null_terminated_bytes` n'y figure pas.** Le message de `v3_unstable_dtype_warning` dit explicitement : « may change without warning… may be unreadable by other Zarr libraries ».
- `create_array(dtype=object)` est refusé : il faut passer `'str'` ou `StringDType()`, ce que le code fait déjà.
- `StringDType` avec `na_object` défini est refusé par zarr (`VariableLengthUTF8`, résolution du dtype).

### pandas 3 (3.1.0)

- `future.infer_string` est actif par défaut : une colonne de chaînes `object` est inférée en `str` (`StringDtype(storage='python', na_value=nan)` sans pyarrow ; `storage='pyarrow'` si pyarrow est installé).
- Une colonne construite depuis un tableau `StringDType` reste en `object` (mesuré) : pandas 3.1 ne mappe pas `StringDType` vers `str`.

## Options pour la décision « Quelle politique de dtype chaîne retenir pour allel2 ? »

1. **Adopter `StringDType` comme dtype chaîne par défaut** (`default_string_dtype` passe de `'object'` à `StringDType()` dans `read_vcf`). Cohérent avec zarr 3 (aller-retour sans conversion), plus rapide que `object`. Changement visible pour les utilisateurs qui testent `dtype == object`. Il faut aussi gérer pandas (convertir en `str` dans `vcf_to_dataframe`).
2. **Garder `object` en mémoire et corriger les tests** pour comparer les valeurs plutôt que les dtypes après aller-retour zarr. Changement minimal pour l'API, mais `read_vcf` et la relecture zarr divergent toujours, et `object` reste lent.
3. **Imposer `S` (octets fixes)** : compact et rapide, mais en zarr v3 cela s'appuie sur un type non spécifié. Ce n'est viable qu'en écrivant du zarr **v2** (ce qui touche la zone floue « format de stockage de `vcf_to_zarr` »).

Pour les tests `vcf_to_dataframe`, quel que soit le choix, l'attente `np.dtype(object)` doit être remplacée par le dtype chaîne de pandas 3.

Point d'attention pour Rust : un tableau `object` ne peut pas être produit ni lu efficacement depuis Rust sans le GIL ; `S` et `StringDType` sont des représentations sans objets Python. Le support de `StringDType` côté rust-numpy relève de la recherche sur le build hybride.

## Non vérifié

- Comportement des lecteurs non-Python (zarrs en Rust, TensorStore) sur le type `string` : spécifié, mais pas testé ici.
- Format exact des zarr Ag3/MalariaGEN existants (v2 `|O` ou `|S`) : non relu ici. D'après le code de zarr, les deux restent lisibles, en `StringDType` et en `S` respectivement.
- Comportement avec pyarrow installé (`storage='pyarrow'`) : non mesuré.
