# Crates Rust pour remplacer le parseur VCF Cython (issue #5)

Question (ticket #5, carte #1) : des crates Rust (noodles-vcf, rust-htslib, autres) peuvent-elles produire les mêmes tableaux que `read_vcf` (sélection de champs, types numériques par champ, Number=A/R/G, ANN, filtres, régions tabix, bgzip) avec de bonnes performances ? Quelles lacunes imposeraient un parseur maison ? Existe-t-il des benchmarks publics comparables ?

Date des recherches : 2026-10-26.

## Réponse courte

Oui pour la **lecture brute** : noodles-vcf (pur Rust) et rust-htslib lisent les enregistrements VCF/BCF, les régions indexées (tabix/CSI) et le BGZF. **Non** pour la **sortie de `read_vcf`** telle quelle : aucune des deux crates ne fournit la couche qui fait la spécificité d'allel2 (résolution de `Number=A/R/G` en formes de tableaux, découpage de `ANN` en sous-champs, typage par champ avec `fills`, nommage `variants/` et `calldata/`, encodage des génotypes). Cette couche est donc à écrire, quelle que soit la crate choisie. Aucun benchmark public comparant noodles, rust-htslib et le Cython d'allel2 n'a été trouvé : il faudra mesurer soi-même.

## Ce que fait allel2 aujourd'hui (référence locale)

Sources : `allel2/io/vcf_read.py` et `allel2/opt/io_vcf_read.pyx` (4 248 lignes), branche `research/rust-vcf`.

- **Signature de `read_vcf`** : `fields`, `exclude_fields`, `rename_fields`, `types`, `numbers`, `alt_number`, `fills`, `region`, `tabix`, `samples`, `transformers`, `buffer_size`, `chunk_length`, `log`.
- **Nombre de valeurs** (`_normalize_number`, `vcf_read.py` l. 1556) : `.` donne 1 ; `A` donne `alt_number` ; `R` donne `alt_number + 1` ; `G` donne **3 en dur** (quel que soit le nombre d'allèles ou la ploïdie) ; un entier est lu tel quel. `DEFAULT_ALT_NUMBER = 3` (l. 39), donc un champ `Number=A` sort avec 3 colonnes par défaut.
- **Types** : int8/16/32/64, uint8/16/32/64, float32/64, flags, chaînes et objets, avec choix par champ via `types`.
- **Fills** : valeurs de remplissage par champ via `fills`.
- **ANN** : `ANNTransformer` (pyx, l. 3898 et suivantes) découpe la chaîne `ANN` en 18 sous-champs `variants/ANN_*` (`ANN_Allele`, `ANN_Annotation`, `ANN_Gene_Name`, `ANN_HGVS_c`, `ANN_cDNA_pos`, etc.), avec les sous-champs `_pos`/`_length` pour `cDNA`, `CDS` et `AA`.
- **Régions** : `_setup_input_stream` (l. 1051) délègue l'accès indexé au **binaire externe `tabix`** via `subprocess` ; sans `tabix`, il **retombe sur un parcours complet** via `gzip.open`. La détection de compression se fait par suffixe (`input.endswith('gz')`), sans lecture de l'en-tête BGZF.
- **Dépendances** : `pyproject.toml` ne dépend d'aucune bibliothèque htslib ; le parseur est autonome en Cython.

## Crates Rust examinées

### noodles (pur Rust, MIT)

Sources : [README du dépôt zaeleus/noodles](https://github.com/zaeleus/noodles) ; [docs.rs noodles-vcf](https://docs.rs/noodles-vcf/latest/noodles_vcf/) ; [docs.rs noodles-tabix](https://docs.rs/noodles-tabix/latest/noodles_tabix/) ; [docs.rs noodles-bgzf](https://docs.rs/noodles-bgzf/latest/noodles_bgzf/) ; [CHANGELOG noodles-vcf](https://github.com/zaeleus/noodles/blob/master/noodles-vcf/CHANGELOG.md).

- **Formats** (README) : VCF 4.3 et 4.4, BCF 2.2, tabix, CSI, BGZF, BAM, CRAM, SAM, FASTA, FASTQ, GFF3, GTF, htsget, refget. Le README indique que l'API est **« still considered experimental »** et ne fait **aucune affirmation de performance**.
- **Versions** observées : noodles-vcf 0.95.0 (2026-10-15 selon le changelog), noodles-tabix 0.68.0, noodles-bgzf 0.52.0. Rust minimal : 1.91 depuis noodles-vcf 0.94.0.
- **Régions** : le changelog indique que depuis 0.83.0, `Query` devient un adaptateur de lecteur ; l'accès indexé passe donc par noodles-tabix/CSI. Le lecteur tabix est documenté pour « find relevant records for a given genomic region », mais la signature exacte de la requête n'a pas été vérifiée sur la page docs.rs.
- **Typage INFO/FORMAT** : le changelog indique des efforts de conformité VCF 4.4/4.5 (ploïdie `P`, formats de comptes locaux `LA`/`LR`/`LS`, `LGL`/`LGP` en float depuis 0.85.0). Le changelog indique aussi que 0.63.0 a **retiré le sérialiseur du `Number` INFO** et que 0.95.0 **rejette les valeurs INFO et sample vides**, ce qui peut casser la lecture de fichiers mal formés.
- **Performance** : seule entrée pertinente du changelog, 0.78.0 : les champs INFO non demandés ne sont pas parsés (la valeur n'est parsée que si la clé correspond). Ce point rejoint le besoin de sélection de champs d'allel2.
- **ANN** : aucune entrée du changelog ne mentionne ANN. Pas de parsing spécifique : la valeur est une chaîne, à découper soi-même.
- **Non vérifié** : la définition exacte de l'énumération `Number` (variantes `A`, `R`, `G`, `Unknown`, etc.) n'a pas pu être lue sur docs.rs ; l'existence des variantes A/R/G est supposée d'après le contexte VCF 4.5 du changelog.

### rust-htslib (liaison C vers htslib, MIT)

Sources : [docs.rs rust-htslib, module bcf](https://docs.rs/rust-htslib/latest/rust_htslib/bcf/) ; [docs.rs bcf::Record](https://docs.rs/rust-htslib/latest/rust_htslib/bcf/record/struct.Record.html) ; [src/bcf/record.rs du dépôt rust-bio/rust-htslib](https://raw.githubusercontent.com/rust-bio/rust-htslib/master/src/bcf/record.rs) ; [issue #464](https://github.com/rust-bio/rust-htslib/issues/464).

- **Version** documentée : 1.0.2 ; dépend de `hts-sys` ^2.2.0 (htslib C embarquée).
- **Lecteurs** : `Reader` (VCF/BCF), `IndexedReader` (accès par région), `Writer`.
- **Accesseurs typés** (vérifiés dans le source) :
  - INFO : `info(tag).integer()` renvoie `Result<Option<...&[i32]>>` ; `float()` renvoie `&[f32]` ; `flag()` renvoie `Result<bool>` ; `string()` renvoie `Option` de vecteur de chaînes. Une étiquette absente donne `Ok(None)`.
  - FORMAT : `format(tag).integer()`, `float()`, `string()` renvoient `Result<...>` **sans `Option`** ; une étiquette absente donne `Err(BcfMissingTag)`. Pas de méthode `flag`.
  - Les valeurs manquantes à l'intérieur des données sont signalées par des sentinelles (`MISSING_INTEGER = i32::MIN`, testées par `is_missing()`).
- **Conséquence** : les entiers sortent en `i32` et les flottants en `f32` ; les types int8/uint8/int64 demandés par `types` impliquent une conversion explicite. Les étiquettes FORMAT absentes sont une erreur à gérer.
- **Régions** : `IndexedReader` existe ; la documentation fetchée ne détaille pas les méthodes `fetch`, qui n'ont donc pas été vérifiées.
- **ANN** : pas de support particulier ; la chaîne est renvoyée brute.
- **Build** : liaison à htslib en C ; le build nécessite une chaîne de compilation C et les bibliothèques de compression (zlib, libdeflate selon les options). Pour un package PyPI multiplateforme, c'est une contrainte supplémentaire par rapport à noodles.
- **Régression connue** : [issue #464](https://github.com/rust-bio/rust-htslib/issues/464), ouverte depuis 2025-01-27, rapporte un ralentissement d'environ ×10 sur Apple Silicon entre rust-htslib 0.48.0 (8,5 s) et 0.49.0 (83,2 s). La mise à jour de hts-sys est suspectée (libdeflate), sans correctif recensé. À surveiller si htslib est retenu, et particulièrement pertinent pour l'environnement macOS de l'équipe.

### Autres pistes

- **oxbow** (Python, s'appuie sur noodles selon des résumés de recherche non vérifiés sur le dépôt) : `VcfFile(source, compressed=False, fields='*', info_fields='*', genotype_fields='*', genotype_by='sample', samples=None, coords='11', regions=None, index=None, batch_size=131072)`, sortie vers Polars/Arrow. Documentation ([oxbow VcfFile](https://oxbow.readthedocs.io/en/latest/api/python/oxbow.core.VcfFile.html)) : **pas de description du typage Arrow des champs INFO/FORMAT** (à lire via `schema` à l'exécution), **pas de mention d'ANN**, **pas de mention explicite de BGZF ou tabix** (seul `compressed` est documenté). Intéressant comme référence de sélection de champs, mais ne couvre pas la sortie d'allel2.
- **polars-bio** : le README fetché n'a pas pu être lu (404) ; aucune information vérifiée sur sa lecture VCF.
- **bio2zarr / vcf2zarr** : la documentation n'a pas pu être lue (page vide) ; non vérifié.
- **Crate `vcf`** (vcf-rs) : dernière version en 2022, sans benchmark ; écartée.

## Comparaison avec les besoins d'allel2

| Besoin de `read_vcf` | noodles-vcf | rust-htslib | Lacune à combler |
|---|---|---|---|
| Sélection de champs INFO/FORMAT | Oui (0.78.0 : skip des champs non demandés) | Oui (accès par étiquette) | Mapping `fields` → nom de tableau `variants/*`, `calldata/*` |
| Types numériques par champ | Partiel (types typés selon l'en-tête, détail non vérifié) | `i32`/`f32` seulement | Conversion vers int8/16/32/64, uint*, float32/64 selon `types` |
| `Number=A/R/G` | Présent côté en-tête (VCF 4.4/4.5), énumération non vérifiée | Brut (pas de forme résolue) | Résolution `alt_number`, `R = alt_number+1`, `G` (convention 3 en dur d'allel2) |
| ANN | Non | Non | Parseur maison des 18 sous-champs, dont `_pos`/`_length` |
| Filtres | Non vérifié | Oui (`filters()`, `has_filter`) | Tableaux booléens par filtre (à construire) |
| Régions tabix/CSI | Oui (noodles-tabix/CSI, sans binaire externe) | `IndexedReader` (fetch non vérifié) | Remplacer le `subprocess tabix` et le repli par parcours complet |
| bgzip/BGZF | Oui (noodles-bgzf) | Oui (htslib) | Détection par en-tête, pas par suffixe |
| Valeurs manquantes et `fills` | Non documenté | Sentinelles i32::MIN | Reproduire les `fills` d'allel2 |
| Sortie en tableaux NumPy | Non | Non | Couche PyO3 à écrire |

## Benchmarks publics

- **Aucun benchmark public** comparant noodles, rust-htslib ou un parseur Rust avec le Cython d'allel2 n'a été trouvé.
- **cyvcf2 (Cython + htslib)** : dans son article [Bioinformatics](https://academic.oup.com/bioinformatics/article/2971439/cyvcf2), un filtre sur le chromosome 22 de 1000 Genomes donne environ 248 s pour cyvcf2 contre 224 s pour bcftools, avec pysam environ 7 fois plus lent et pyvcf environ 168 fois plus lent. Source lue via un résumé de recherche, non relue en entier. Pertinent comme ordre de grandeur, pas comme comparaison directe.
- **Benchmark vcfv** (Python/Polars, [PyPI](https://pypi.org/project/vcfv/)) : 500 000 variants, 10 exécutions, moyenne tronquée ; ne concerne pas un parseur Rust.

Il faudra donc mesurer soi-même, sur les mêmes fichiers (par exemple un chunk Ag1000G), avec `io_vcf_read.pyx` comme référence.

## Recommandation

1. **Le choix par défaut est noodles** : pur Rust, pas de dépendance C, build simple pour des wheels via maturin/PyO3, et lecture indexée sans binaire `tabix`. Le risque est la maturité (API déclarée expérimentale, changements cassants fréquents).
2. **rust-htslib** reste une option de repli si noodles est trop lent sur les mesures, mais il impose htslib en C et une régression non résolue sur macOS.
3. **Dans les deux cas**, il faut écrire et tester soi-même : le découpage ANN, la résolution `Number`, les conversions de types, les `fills`, et la construction des génotypes. Ce code porte la compatibilité avec `read_vcf` et doit être validé par comparaison bit à bit avec la sortie Cython.
4. **Prototype avant tout portage** : suivre la règle de la carte #1 (porter seulement si le profilage désigne le parseur comme goulot et si le prototype dépasse ×2 face au Cython). Le prototype doit être mesuré sur un chunk réel, avec la même sélection de champs.

## Limites de cette recherche

- Pages docs.rs non lues : énumération `Number` de noodles, méthodes `fetch` de rust-htslib.
- Pages non accessibles : README polars-bio, bio2zarr (documentation vide), dépôt vcfv (404).
- Aucune mesure de performance réalisée ici.
