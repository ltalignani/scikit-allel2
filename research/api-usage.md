# Usage de l'API `allel` dans le code aval (inventaire pour élagage)

Issue : ltalignani/scikit-allel2 #2. Date des mesures : 2026-10-26.
Périmètre : API publique de `allel2` (fork de cggh/scikit-allel) vue depuis les usages aval, surtout `malariagen_data`.

## Réponse courte

- **Le backend chunked est-il utilisé ?** Pas par `malariagen_data` (commit `a69b0f4`, 2026-10-01) : zéro occurrence de `allel.chunked`, `GenotypeChunkedArray`, `vcf_to_zarr` ou `vcf_to_hdf5` dans le package. Il reste utilisé par des pipelines et notebooks MalariaGEN (historiques, et quelques pipelines actifs) et par des projets publics. Verdict : **marginal mais non nul**, à ne pas supprimer sans migration de ces usages.
- **Dask** (`GenotypeDaskArray`, `GenotypeDaskVector`) : **très utilisé**, c'est le cœur de `malariagen_data` (20 sites d'appel).
- **Stockage** : `vcf_to_zarr` est utilisé en production (2 scripts Docker de `malariagen/pipelines`). `vcf_to_hdf5`, `vcf_to_npz`, `read_vcf` et consorts ne sont **pas** utilisés dans les dépôts MalariaGEN clonés.
- **hmmlearn / protopunica** : `roh_mhmm` (hmmlearn) n'apparaît que dans 1 notebook. `roh_poissonhmm` (protopunica) n'a aucun appel aval. `malariagen_data` réimplémente lui-même la HMM protopunica et importe deux helpers internes d'allel (`allel.stats.roh._hmm_derive_transition_matrix`, `allel.stats.misc.tabulate_state_blocks`). Ces deux helpers ne dépendent pas de hmmlearn/protopunica et doivent être conservés.

Point de vigilance : `malariagen_data` dépend du paquet upstream `scikit-allel>=1.3.13` (`pyproject.toml`), pas de `allel2`. Les usages mesurés sont donc ceux de l'API upstream, que `allel2` doit rester capable de servir.

## Méthode et limites

**Sources primaires (lecture seule, clones superficiels dans `/tmp/allel-research/`) :**

| Dépôt | Commit | Date du commit |
|---|---|---|
| malariagen/malariagen-data-python | a69b0f4 | 2026-10-01 |
| malariagen/pipelines | e88dd12 | 2024-01-22 |
| malariagen/funestus_popgen_ms | bc9db80 | 2025-06-30 |
| malariagen/ano-spp (fichier lu via API) | non cloné | |
| malariagen/methods-dev | 6839262 | 2018-07-19 |
| malariagen/vector-tools | 9b47188 | 2019-09-03 |
| malariagen/ag1000g-phase3-data-paper | a88a56f | 2021-03-25 |

Comptage : grep de `allel.<symbole>` (et alias `allel.chunked.*`, `allel.stats.*`) sur ces clones. Le comptage est un comptage de fichiers, pas de méthodes d'objets. Les appels de méthodes (`.count_alleles()`, `.is_het()`, `.to_haplotypes()`, `.to_n_ref()`, etc.) ne sont pas capturés par cette méthode.

**Recherche de code GitHub** (`gh api search/code`, `gh search code`, 2026-10-26) :
- `"<symbole>" language:python` et la variante `... -repo:cggh/scikit-allel -repo:ltalignani/scikit-allel2 -repo:malariagen/malariagen-data-python` (script `gh_counts.py` puis `gh_counts_ext.py`).
- `allel.vcf_to_zarr`, `allel.vcf_to_hdf5`, `allel.GenotypeChunkedArray` (25–30 premiers résultats listés).
- `"import allel" --owner malariagen` (99 résultats, plafonné).

**Limites de la recherche de code :**
- Le `total_count` est une estimation GitHub, pas un décompte exact. Les tokens sont mal séparés : `read_vcf` matche presque tout (3184), donc inutilisable.
- Les résultats sont plafonnés (~1000) et la recherche est limitée en débit (10 requêtes/min). Les forks sont exclus par défaut.
- Les comptes incluent des notebooks avec sorties HTML (ex. `allel.chunked.storage_zarr.ZarrTable` apparaît dans les sorties de `methods-dev`), ce qui gonfle certains comptes.

**Autres limites :**
- Dépôts MalariaGEN non clonés : `vobs-sea-pilot-paper`, `datalab`, `agam-selection-atlas`, `agam-report-base`, `pv-muhap`, `kestrel`, rapports Ag1000G phase 1/2 et une partie de `ano-spp`. Le dépôt `ano-spp` n'est couvert que par un fichier lu via API.
- Les dépôts privés, les notebooks non publiés et les usages dans sgkit ne sont pas couverts.
- Côté allel2, l'inventaire a été fait sur le code local (`allel2/__init__.py`, `allel2/model/*`, `allel2/io/*`, `allel2/stats/*`, `allel2/chunked/*`), pas sur la doc `scikit-allel.readthedocs.io` (non consultée).

**Incident d'environnement :** un hook de lint (`lint-on-write.sh`) a créé un venv `.venv/` à la racine du worktree lors d'une écriture de script en `/tmp`. Son installation a échoué mais le dossier `.venv/` existe toujours (non suivi car il contient son propre `.gitignore`). La suppression a été refusée par le classifieur de permissions. **À supprimer manuellement** : `/Users/loictalignani/scikit-allel-master/.claude/worktrees/agent-ad1023c275f9d2940/.venv`.

**Règle de classification** (jugement appliqué, à valider) :
- **très utilisé** : cœur de `malariagen_data` ou ≥ 5 fichiers dans les dépôts MalariaGEN clonés, avec usage actif (2024 et après, ou dans `malariagen_data`).
- **marginal** : 1 à 4 fichiers, ou usage uniquement historique (2016–2021), ou uniquement dans des projets publics.
- **jamais vu** : 0 fichier dans les dépôts MalariaGEN clonés. Le statut public est indiqué à part quand il est mesuré.

## Inventaire classé

Notation : `mgd` = malariagen-data-python ; `pipes` = malariagen/pipelines ; `fun` = funestus_popgen_ms ; `mdev` = methods-dev ; `vt` = vector-tools ; `ag3p` = ag1000g-phase3-data-paper ; `anospp` = ano-spp.

### Modèles de données (`allel.model.ndarray`, `allel.model.chunked`, `allel.model.dask`)

| Symbole / module | Statut | Preuves |
|---|---|---|
| `GenotypeArray` | **très utilisé** | `mgd` 10 appels (`util.py:1668`, `anopheles.py`, `dipclust.py:299`, `hapclust.py:665`, `snp_frq.py:221`, `g123.py:99`, ...). 24 fichiers sur 4 dépôts clonés. Hits publics bruts : 924 (non filtré). |
| `GenotypeDaskArray` (via `allel.model.dask`) | **très utilisé** | `mgd` 18 appels (`anopheles.py:829,1223`, `ld.py:110`, `hapclust.py:322,869`, `h12.py:55,257`, `xpehh.py:180-181`, `snp_data.py:1439,2265`, ...). 29 fichiers sur 5 dépôts clonés. Hits publics hors cœur : 65. |
| `GenotypeDaskVector` | **très utilisé** dans `mgd` (usage ponctuel) | `mgd/heterozygosity.py:368,485` (ROH par HMM). 2 fichiers. Hits publics : 3. |
| `GenotypeChunkedArray` (`allel.model.chunked`) | **marginal** (historique + quelques pipelines actifs) | `mgd` : 0. `vt` : `pipelines/phasing-by-transmission/Snakefile` (2019) et `scripts/evaluate_phasing_performance.py`. `anospp` : `work/8_ag1k_analysis/1_diversity.ipynb` (notebook `ampl_snps = allel.GenotypeChunkedArray(...)`). `mdev` : ~10 notebooks Pf6.0 (2016–2018). Hits publics hors cœur : 73, par ex. `hardingnj/xpclr`, `stsmall/Wb_sWGA`, `NinaMercedes/PopGen`. |
| `HaplotypeArray` | **marginal** (direct) | `mgd/hapclust.py:327` ; `vt` 2 appels. Utilisé surtout indirectement via `GenotypeArray.to_haplotypes()` (`hapclust.py:665`). Hits publics bruts : 213. |
| `AlleleCountsArray` | **très utilisé** (indirect) | `mgd/anopheles.py:224`, `snp_data.py:1702,1975` ; `ag3p` 3 fichiers. Hits publics bruts : 83. |
| `SortedIndex` | **très utilisé** | `mgd` (`util.py:778`, `hapclust.py:666`, `karyotype.py:98`) ; `ag3p` 8 appels ; `vt` 8 appels. 12 fichiers. |
| `FeatureTable` (`from_gff3`) | **marginal** | `fun` : 2 notebooks `find_cas9_targets_*` (`FeatureTable.from_gff3`). |
| `VariantTable`, `UniqueIndex`, `SortedMultiIndex`, `ChromPosIndex` | **jamais vu** | 0 fichier dans les dépôts clonés. |
| `GenotypeVector` (classe nue) | **jamais vu** | 0 fichier. `GenotypeDaskVector` est en revanche utilisé. |
| `GenotypeAlleleCounts*` | **jamais vu** | 0 fichier. |
| `HaplotypeChunkedArray`, `AlleleCountsChunkedArray`, `AlleleCountsChunkedTable` | **marginal** / **jamais vu** | `vt` : 1 usage de `AlleleCountsChunkedArray`. Les deux autres : 0. |
| `VariantChunkedTable` | **marginal** (historique) | `mdev` 4 fichiers (2016–2018). |
| `allel.chunked` (wrapper `ChunkedArrayWrapper`, `storage_zarr.ZarrTable`) | **marginal** (historique) | `mdev` : `ZarrTable` dans 4 fichiers (2016–2018, sorties ipynb incluses). `vt` : `allel.chunked.ChunkedArrayWrapper` 1 fois. Hits publics bruts `allel.chunked` : 13 hors cggh. |
| Stockages h5py / zarr de `allel.chunked` (`HDF5MemStorage`, `ZarrMemStorage`, `hdf5_cache`, etc.) | **jamais vu** en direct | 0 appel direct dans les dépôts clonés (`hdf5_cache` : 0). Les stockages ne sont utilisés que via `vcf_to_zarr`/`vcf_to_hdf5` et les classes chunked. |
| `allel.model.dask` (module entier) | **très utilisé** | Accès via l'export top-level `allel.GenotypeDaskArray` / `GenotypeDaskVector` (`allel2/__init__.py` importe `model.dask` sous condition `dask` installé). Hits publics `allel.model.dask` : 6. |
| `allel.abc.ArrayWrapper` | **marginal** (historique) | `mdev` 13 occurrences (2016–2018). |

### I/O (`allel.io`)

| Symbole / module | Statut | Preuves |
|---|---|---|
| `vcf_to_zarr` | **marginal** (mais en production) | `pipes/dockerfiles/CohortVcfToZarr/cohort_vcf_to_zarr.py:142` et `SampleVcfToZarr/sample_vcf_to_zarr.py:171` (2021, 2022). `mdev` : 1 notebook `pf3k_techbm/.../20180717_SR_vs_LR_analysis.ipynb`. Hits publics bruts : 148 ; sur un échantillon de 30 : `kr-colab/locator`, `genxnetwork/grape`, `frankvogt/vcf2gwas`, `DRL/farrago`, `travc/*`, `David-Peede/*`, etc. |
| `vcf_to_hdf5` | **jamais vu** dans MalariaGEN ; **marginal** public | 0 fichier dans `mgd`, `pipes`, `fun`, `mdev`, `vt`, `ag3p`, `anospp`. Hits publics bruts : 87 ; exemples : `mathiesonlab/pg-gan` (`vcf2hdf5.py`), `kr-colab/ReLERNN`, `ellisztamas/1001crosses` (`vcf_to_HDF5.py`), `santiago1234/popgene-pipelines`. |
| `vcf_to_npz` | **jamais vu** dans MalariaGEN ; **marginal** public | 0 fichier. Hits publics bruts : 11. |
| `read_vcf` | **jamais vu** dans MalariaGEN ; usage public **non mesurable** | 0 fichier. Hits publics bruts : 3184 (bruit de tokenisation). Décision à prendre sur un autre critère que cette recherche. |
| `read_vcf_headers`, `vcf_to_dataframe`, `vcf_to_csv`, `vcf_to_recarray`, `iter_vcf_chunks` | **jamais vu** | 0 fichier. |
| `write_vcf`, `write_fasta` | **jamais vu** | 0 fichier. |
| `gff3_to_dataframe`, `gff3_to_recarray`, `iter_gff3` | **marginal** | `ag3p` et `mdev` (`gff3_to_dataframe`, 2 fichiers). |
| `allel.io.util` (`array_to_hdf5`, `recarray_to_hdf5_group`, etc.) | **jamais vu** | 0 fichier. |

### Statistiques (`allel.stats.*`)

| Symbole / module | Statut | Preuves |
|---|---|---|
| `moving_statistic` | **très utilisé** | `mgd` 18 appels (`anopheles.py:876`, `h12.py:61`, `g123.py:144`, `xpehh.py:228`, `fst.py:110`, `heterozygosity.py:383`, ...) ; `fun` 2. 12 fichiers. Hits publics bruts : 241. |
| `windowed_statistic` | **utilisé** | `ag3p` 2, `fun` 1, `mdev` (occurrences brutes, surtout dans des sorties de notebooks). 7 fichiers au total. |
| `position_windows`, `equally_accessible_windows` | **marginal** | `ag3p` (`position_windows`, `equally_accessible_windows`), `vt` (`position_windows`). |
| `window_locations`, `per_base` | **jamais vu** | 0 fichier. |
| `index_windows` | **marginal** | `mgd/h1x.py:419`. |
| `windowed_count` | **marginal** (historique) | `mdev` 3 appels. |
| `mean_pairwise_difference` | **marginal** | `mgd/anopheles.py:243`, `fun` 2 appels. |
| `sequence_diversity`, `watterson_theta`, `tajima_d`, `windowed_tajima_d`, `windowed_diversity` | **marginal** (`windowed_tajima_d` historique) | `anospp/1_diversity.ipynb` (`sequence_diversity`, `watterson_theta`, `tajima_d`) ; `mdev` `windowed_tajima_d` 4 fichiers. |
| `ihs`, `xpehh`, `standardize_by_allele_count` | **marginal** (mais fonctionnalités exposées) | `mgd/anopheles.py:846,867,876` (`ihs`, `standardize_by_allele_count`) ; `mgd/xpehh.py:203` (`xpehh`). 2 fichiers. Hits publics : `ihs` 46, `xpehh` 30 (non filtré). |
| `garud_h`, `moving_garud_h` | **jamais vu** dans MalariaGEN (`mgd` réimplémente `_garud_h12`/`_garud_g123`) ; **marginal** dans `fun` | `fun/analysis/H12_selection_scans/compute_H12_statistic.ipynb:950,1028` (`moving_garud_h`). |
| `nsl`, `xpnsl`, `ehh_decay`, `voight_painting`, `haplotype_diversity`, `moving_haplotype_diversity`, `moving_delta_tajima_d`, `pbs` | **jamais vu** | 0 fichier. |
| `pca` (`stats.decomposition.pca`) | **très utilisé** | `mgd/anoph/pca.py:352`, `fun`, `ag3p`, `mdev` (`allel.stats.pca`). 7 fichiers. Hits publics bruts : 202. |
| `GenotypePCA` | **marginal** | `mgd/anoph/pca.py:356`. |
| `randomized_pca`, `GenotypeRandomizedPCA`, `StandardScaler`, `CenterScaler`, `PattersonScaler` | **jamais vu** | 0 fichier. |
| `blockwise_hudson_fst` | **utilisé** | `mgd/anoph/fst.py:457`, `mdev` (5 fichiers au total avec `mgd`). |
| `moving_hudson_fst` | **marginal** | `mgd/anoph/fst.py:107`. |
| `hudson_fst`, `windowed_hudson_fst`, `average_hudson_fst` | **marginal** (`hudson_fst` dans `fun`) | `fun` 1 notebook. Hits publics : `hudson_fst` 115. |
| `weir_cockerham_fst`, `patterson_fst`, `*_patterson_*`, `average_*` | **jamais vu** dans MalariaGEN | 0 fichier. Hits publics bruts : `weir_cockerham_fst` 77. |
| `locate_unlinked` | **marginal** (mais en production) | `mgd/anoph/ld.py:114` ; `ag3p` 1. Hits publics bruts : 53. |
| `rogers_huff_r`, `windowed_r_squared`, `plot_pairwise_ld` | **jamais vu** | 0 fichier. |
| `heterozygosity_observed`, `heterozygosity_expected`, `inbreeding_coefficient` | **jamais vu** | 0 fichier (`mgd/heterozygosity.py` réimplémente sa logique). |
| `sfs`, `sfs_folded_scaled`, `joint_sfs`, `plot_joint_sfs`, toute la famille `sf` | **marginal** (historique) | `mdev` 4 fichiers (2016–2018). Hits publics bruts : `sfs` 104. |
| `condensed_coords` | **marginal** | `ag3p` 1. |
| `pairwise_distance`, `pcoa`, `plot_pairwise_distance` | **jamais vu** | 0 fichier. |
| `mendel_errors`, `phase_by_transmission`, `paint_transmission` | **marginal** | `vt` (pipeline de phasage 2019) ; `mendel_errors` 1 fichier. |
| `tabulate_state_transitions`, `create_allele_mapping` | **marginal** | `vt` 1 fichier chacun. |
| `tabulate_state_blocks` (`stats.misc`) | **marginal** mais critique | `mgd/anoph/heterozygosity.py:42`, import interne dans une fonction HMM. |
| `roh_mhmm` (**hmmlearn**) | **marginal** | `fun/analysis/ROH/compute_ROH.ipynb:413` (1 notebook). Hits publics bruts : 9. |
| `roh_poissonhmm` (**protopunica**) | **jamais vu** dans MalariaGEN (appel direct) | 0 fichier. Hits publics bruts : 6. |
| `_hmm_derive_transition_matrix` (`stats.roh`, privé) | **marginal** mais critique | `mgd/anoph/heterozygosity.py:43`. Pure numpy, sans dépendance hmmlearn/protopunica. |
| Import direct de `protopunica` | Dépendance réelle de `mgd` | `mgd/anoph/heterozygosity.py:48` (`from protopunica import HiddenMarkovModel, PoissonDistribution`) ; `mgd/pyproject.toml:38` déclare `protopunica>=0.14.8.post2` ; `allel2/pyproject.toml` `[optional]` le déclare aussi. |
| `plot_*` (`plot_sfs`, `plot_variant_locator`, `plot_voight_painting`, `plot_haplotype_frequencies`, etc.) | **jamais vu** sauf `plot_joint_sfs` | `mdev` 4 fichiers historiques. |
| `hw`, `admixture` (`patterson_f2`, `patterson_f3`, `patterson_d`, blockwise, moving) | **jamais vu** | 0 fichier. |
| `mendel`, `misc` | voir plus haut | |

### Autres

| Symbole / module | Statut | Preuves |
|---|---|---|
| `allel.opt` (Cython) | Non mesuré (interne) | Compilé via `allel2/opt/*.pyx`, pas d'API publique. |
| `allel.util`, `allel.model.util` | Non mesuré | Non cherché individuellement. |

## Sources

- malariagen/malariagen-data-python @ a69b0f4 : https://github.com/malariagen/malariagen-data-python/tree/a69b0f4992039a66f9624c0d16433dc7d0dfdf28
  - `malariagen_data/anopheles.py` : https://github.com/malariagen/malariagen-data-python/blob/a69b0f4992039a66f9624c0d16433dc7d0dfdf28/malariagen_data/anopheles.py
  - `malariagen_data/anoph/heterozygosity.py` : https://github.com/malariagen/malariagen-data-python/blob/a69b0f4992039a66f9624c0d16433dc7d0dfdf28/malariagen_data/anoph/heterozygosity.py
  - `pyproject.toml` : https://github.com/malariagen/malariagen-data-python/blob/a69b0f4992039a66f9624c0d16433dc7d0dfdf28/pyproject.toml
- malariagen/pipelines @ e88dd12 : https://github.com/malariagen/pipelines/blob/e88dd12260c0412e8d61904f15cdc8835c50da1c/dockerfiles/CohortVcfToZarr/cohort_vcf_to_zarr.py ; https://github.com/malariagen/pipelines/blob/e88dd12260c0412e8d61904f15cdc8835c50da1c/dockerfiles/SampleVcfToZarr/sample_vcf_to_zarr.py
- malariagen/vector-tools @ 9b47188 : https://github.com/malariagen/vector-tools/blob/9b471887ebb7809cffe894fa678108ee6a9f7921/pipelines/phasing-by-transmission/Snakefile
- malariagen/ano-spp : https://github.com/malariagen/ano-spp/blob/main/work/8_ag1k_analysis/1_diversity.ipynb (fichier lu via `gh api`, branche par défaut)
- malariagen/methods-dev @ 6839262 : https://github.com/malariagen/methods-dev/tree/68392628a6be2287f5c0294fd695912f016ab0f1
- malariagen/funestus_popgen_ms @ bc9db80 : https://github.com/malariagen/funestus_popgen_ms/blob/bc9db80f96fb25d61d75a8b16c8490fb3f8947f7/analysis/ROH/compute_ROH.ipynb
- malariagen/ag1000g-phase3-data-paper @ a88a56f : https://github.com/malariagen/ag1000g-phase3-data-paper/tree/a88a56f999b2e85a1ace755d0564911e5e847311
- Exemples publics (recherche de code, non vérifiés ligne à ligne) : https://github.com/kr-colab/locator/blob/main/scripts/vcf_to_zarr.py ; https://github.com/mathiesonlab/pg-gan/blob/main/vcf2hdf5.py ; https://github.com/hardingnj/xpclr/blob/main/xpclr/util.py
- allel2 local (ce dépôt) : `allel2/__init__.py`, `allel2/model/{ndarray,chunked,dask}.py`, `allel2/io/vcf_read.py` (`vcf_to_npz` l.374, `vcf_to_hdf5` l.605, `vcf_to_zarr` l.860), `allel2/stats/{roh,misc,decomposition,window,fst,ld,selection,sf}.py`, `allel2/chunked/*.py`, `pyproject.toml`, `requirements*.txt`.

## Points d'incertitude

1. **Comptes publics** : les totaux GitHub sont des estimations (tokenisation, plafond, forks exclus). Les chiffres sont à lire comme des ordres de grandeur. `read_vcf` (3184) et `read_vcf_headers` ne sont pas interprétables.
2. **Méthodes non comptées** : l'usage des méthodes d'objets (`GenotypeArray.count_alleles`, `GenotypeDaskArray.is_het`, `.to_haplotypes()`, `.to_n_ref()`, etc.) n'est pas mesuré par le grep de symboles. Un symbole "jamais vu" peut être utilisé via une méthode.
3. **Périmètre des dépôts** : certains dépôts MalariaGEN ne sont pas clonés (voir plus haut). Un "jamais vu" signifie "jamais vu dans les dépôts examinés", pas "jamais utilisé".
4. **Version upstream vs allel2** : les usages mesurés visent l'API upstream (`scikit-allel`), pas `allel2`. Les différences de comportement ou de signature entre upstream et `allel2` ne sont pas vérifiées ici.
5. **Dates** : `methods-dev` (2018), `vector-tools` (2019) et `ag3p` (2021) sont historiques. Les pipelines `cohort_vcf_to_zarr.py` / `sample_vcf_to_zarr.py` (2021–2022) sont les usages les plus récents de `vcf_to_zarr` vus ici. L'état de production réel n'est pas vérifié.
6. **Notebooks** : les notebooks avec sorties gonflent certains comptes. Les décomptes par fichier sont plus fiables que les occurrences brutes.
7. **Symboles non mesurés individuellement** : `allel.util`, `allel.model.util`, `allel.hw`/`allel.admixture` côté public, `allel.plot_*` côté public, et les variantes `_chunked` de fst/diversity.
