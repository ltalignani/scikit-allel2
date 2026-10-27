---
status: accepted
---

# Parseur VCF en Rust, avec repli sur le Cython

Le parseur VCF Cython (`VCFChunkIterator`) occupe 90 à 99 % du temps de `read_vcf` et de `vcf_to_zarr` sur nos données. Il ne travaille que sur un fil, et la décompression n'y compte que pour une petite part. Le 2026-10-27, nous avons décidé de **réécrire ce parseur en Rust** (PyO3, rayon), derrière les mêmes fonctions publiques.

Le parsing des lignes est écrit à la main. La décompression BGZF multi-fil et les requêtes par région passent par les crates **noodles** (`noodles-bgzf`, `noodles-tabix`/`csi`), au lieu du binaire `tabix`. Le parseur Rust couvre tous les types de champs (colonnes fixes, FILTER, INFO et FORMAT typés, Number=1/A/R/G/n, valeurs de remplissage, chaînes `StringDType`). Les autres cas **retombent sur le parseur Cython** : `transformers` (ANN), entrée de type objet fichier Python, type ou `Number` non encore géré. Le Cython est retiré lors d'une version majeure, une fois la parité atteinte.

Un prototype limité à GT mesure ×6,4 sur un fil et ×16 sur 10 fils sur le chr22 de 1000 Genomes. Sur une région Ag2 lue par `tabix`, il mesure ×2,1 et ×2,8, plafonné par la décompression mono-fil de `tabix`. Ses sorties sont identiques à celles du Cython.

## Considered Options

- **Garder le Cython** : rejeté. Le gain mesuré dépasse largement le seuil ×2 fixé pour tout portage.
- **S'adosser entièrement à une crate (noodles, rust-htslib)** : rejeté. Aucune ne produit la sortie de `read_vcf` (types par champ, Number=A/R/G, valeurs de remplissage, génotypes en tableaux), et leur couche d'enregistrements coûterait plus qu'un parsing direct des octets. noodles ne sert qu'aux entrées et sorties.
- **Viser la parité complète avant de livrer** : rejeté. ANN et les transformers bloqueraient le gain sur les usages courants, alors que le repli les garde fonctionnels.

## Consequences

- Deux parseurs coexistent pendant la transition. Un test de non-régression compare leurs sorties, qui doivent être identiques au bit près pour les entiers, sur les VCF de test et sur une région du jeu de benchmark. La variable `ALLEL2_BACKEND=cython` force l'ancien parseur pour comparer ou contourner un bogue.
- Le build requiert une toolchain Rust (setuptools-rust, à côté des extensions Cython) dans la CI et pour toute installation depuis le sdist. Il faut `debug=False` : sans cela, les installations éditables compilent en debug, et le Rust est alors plus lent que le Cython.
- Les régions ne dépendent plus du binaire `tabix` ni de sa sortie d'erreur.
- `read_vcf`, `vcf_to_zarr` et les autres lecteurs reçoivent `n_threads=None` : tous les cœurs disponibles, en respectant `OMP_NUM_THREADS` ou l'affinité CPU.

Détail : [Quels noyaux porter en Rust, et dans quel ordre ?](https://github.com/ltalignani/scikit-allel2/issues/11) et [Prototype Rust d'un premier noyau, mesuré face au Cython](https://github.com/ltalignani/scikit-allel2/issues/10).
