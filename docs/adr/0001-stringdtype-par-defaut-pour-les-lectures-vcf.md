---
status: accepted
---

# StringDType par défaut pour les lectures VCF

scikit-allel renvoyait les champs chaînes des VCF en `object`. Sous zarr 3 et pandas 3, cela ne tient plus. zarr 3 refuse d'écrire du `object` et relit toujours ses chaînes en `numpy.dtypes.StringDType`, y compris les anciens zarr scikit-allel. pandas 3 convertit en `str`. Nous avons donc décidé (2026-10-26) que `read_vcf` et ses dérivés renvoient `StringDType()` par défaut, sans `na_object`. Le parseur Cython produit toujours `object`, puis chaque bloc est converti, pour un coût d'environ 30 à 60 ms par million de valeurs. L'ancien comportement reste disponible champ par champ via `types={'CHROM': object}`.

## Considered Options

- **Garder `object`** : rejeté. `read_vcf` et la relecture zarr divergeraient en permanence, et `object` est 10 à 25 fois plus lent sur le tri et `np.unique`.
- **Octets fixes `S`** : rejeté. Les utilisateurs manipuleraient des `bytes` au lieu de `str`, et en zarr v3 le type `null_terminated_bytes` n'a pas de spécification.

## Consequences

- Changement visible : le code qui teste `dtype == object` casse ; `==`, `np.unique` et `ChromPosIndex` se comportent à l'identique.
- Tout test `dtype.kind == 'O'` du code doit aussi accepter `'T'` (le code de `StringDType`).
- `vcf_to_dataframe` convertit explicitement les chaînes en `str`, car pandas 3.1 laisse un tableau `StringDType` en `object`.
- Le périmètre se limite aux lectures VCF. `UniqueIndex`, les lecteurs GFF et FASTA gardent `object`.

Détail : [Quelle politique de dtype chaîne retenir pour allel2 ?](https://github.com/ltalignani/scikit-allel2/issues/7)
