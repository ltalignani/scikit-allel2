---
status: accepted
---

# zarr v3 par défaut pour vcf_to_zarr

scikit-allel écrivait du zarr v2. Le 2026-10-26, nous avons décidé que `vcf_to_zarr` écrive du **zarr v3 par défaut**, avec un paramètre `zarr_format=` qui permet encore `zarr_format=2`. Le v3 est le format natif de zarr 3, la seule version que `allel2` supporte (`zarr >= 3.1`). Ses types de chaînes (`string`) sont spécifiés dans zarr-extensions. Le mainteneur, principal utilisateur de `vcf_to_zarr`, a préféré s'aligner tout de suite plutôt que de reporter la bascule à une version majeure ultérieure.

## Considered Options

- **v2 par défaut, v3 sur demande** : rejeté. C'était le choix le plus compatible avec scikit-allel et les lecteurs zarr 2, mais il repoussait une migration inévitable et gardait par défaut un format hérité.

## Consequences

- **zarr 2 ne lit pas les fichiers produits par défaut.** Les outils encore en zarr 2 doivent recevoir des fichiers écrits avec `zarr_format=2`. À annoncer dans les notes de version.
- **Les compresseurs numcodecs ne sont pas acceptés tels quels en v3.** `compressor` traduit `numcodecs.Blosc` et `numcodecs.Zstd` vers `zarr.codecs.BloscCodec` et `zarr.codecs.ZstdCodec`, et lève une erreur explicite pour tout compresseur intraduisible. Le défaut historique (Blosc lz4, niveau 5, shuffle) est conservé.
- **Les octets fixes `S` deviennent `null_terminated_bytes` en v3**, un type sans spécification. Pour que le chemin par défaut n'en produise pas, les champs VCF `Character` passent en `StringDType` (voir l'ADR 0001). Un `S` demandé explicitement via `types=` est écrit tel quel, avec l'avertissement de zarr.

Détail : [Quel format zarr vcf_to_zarr écrit-il par défaut, et comment stocke-t-il les octets fixes ?](https://github.com/ltalignani/scikit-allel2/issues/13)
