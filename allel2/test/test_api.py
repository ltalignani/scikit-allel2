# -*- coding: utf-8 -*-
def test_public_api():
    # The idea of this test is to ensure that all functions we expect to be in the
    # public API under the correct namespace are indeed there.

    import allel2

    # allel2.model.ndarray
    assert callable(allel2.GenotypeVector)
    assert callable(allel2.GenotypeArray)
    assert callable(allel2.HaplotypeArray)
    assert callable(allel2.AlleleCountsArray)
    assert callable(allel2.GenotypeAlleleCountsVector)
    assert callable(allel2.GenotypeAlleleCountsArray)
    assert callable(allel2.SortedIndex)
    assert callable(allel2.UniqueIndex)
    assert callable(allel2.SortedMultiIndex)
    assert callable(allel2.VariantTable)
    assert callable(allel2.FeatureTable)

    # allel2.model.dask
    assert callable(allel2.GenotypeDaskVector)
    assert callable(allel2.GenotypeDaskArray)
    assert callable(allel2.HaplotypeDaskArray)
    assert callable(allel2.AlleleCountsDaskArray)
    assert callable(allel2.GenotypeAlleleCountsDaskVector)
    assert callable(allel2.GenotypeAlleleCountsDaskArray)

    # allel2.model.chunked
    assert callable(allel2.GenotypeChunkedArray)
    assert callable(allel2.HaplotypeChunkedArray)
    assert callable(allel2.AlleleCountsChunkedArray)
    assert callable(allel2.GenotypeAlleleCountsChunkedArray)
    assert callable(allel2.VariantChunkedTable)
    assert callable(allel2.AlleleCountsChunkedTable)

    # allel2.model.util
    assert callable(allel2.create_allele_mapping)
    assert callable(allel2.locate_fixed_differences)
    assert callable(allel2.locate_private_alleles)
    assert callable(allel2.sample_to_haplotype_selection)

    # allel2.io.fasta
    assert callable(allel2.write_fasta)

    # allel2.io.gff
    assert callable(allel2.iter_gff3)
    assert callable(allel2.gff3_to_recarray)
    assert callable(allel2.gff3_to_dataframe)
    assert callable(allel2.gff3_parse_attributes)

    # allel2.io.vcf_read
    assert callable(allel2.read_vcf)
    assert callable(allel2.vcf_to_npz)
    assert callable(allel2.vcf_to_hdf5)
    assert callable(allel2.vcf_to_zarr)
    assert callable(allel2.iter_vcf_chunks)
    assert callable(allel2.read_vcf_headers)
    assert callable(allel2.vcf_to_dataframe)
    assert callable(allel2.vcf_to_csv)
    assert callable(allel2.vcf_to_recarray)

    # allel2.io.vcf_write
    assert callable(allel2.write_vcf)
    assert callable(allel2.write_vcf_header)
    assert callable(allel2.write_vcf_data)

    # allel2.stats.admixture
    assert callable(allel2.patterson_f2)
    assert callable(allel2.patterson_f3)
    assert callable(allel2.patterson_d)
    assert callable(allel2.moving_patterson_f3)
    assert callable(allel2.moving_patterson_d)
    assert callable(allel2.average_patterson_f3)
    assert callable(allel2.average_patterson_d)
    # backwards compatibility
    assert callable(allel2.blockwise_patterson_f3)
    assert callable(allel2.blockwise_patterson_d)

    # allel2.stats.decomposition
    assert callable(allel2.pca)
    assert callable(allel2.randomized_pca)

    # allel2.stats.distance
    assert callable(allel2.pairwise_distance)
    assert callable(allel2.pairwise_dxy)
    assert callable(allel2.pcoa)
    assert callable(allel2.condensed_coords)
    assert callable(allel2.condensed_coords_within)
    assert callable(allel2.condensed_coords_between)
    assert callable(allel2.plot_pairwise_distance)

    # allel2.stats.diversity
    assert callable(allel2.mean_pairwise_difference)
    assert callable(allel2.mean_pairwise_difference_between)
    assert callable(allel2.sequence_diversity)
    assert callable(allel2.sequence_divergence)
    assert callable(allel2.windowed_diversity)
    assert callable(allel2.windowed_divergence)
    assert callable(allel2.windowed_df)
    assert callable(allel2.watterson_theta)
    assert callable(allel2.windowed_watterson_theta)
    assert callable(allel2.tajima_d)
    assert callable(allel2.windowed_tajima_d)
    assert callable(allel2.moving_tajima_d)

    # allel2.stats.fst
    assert callable(allel2.weir_cockerham_fst)
    assert callable(allel2.hudson_fst)
    assert callable(allel2.patterson_fst)
    assert callable(allel2.windowed_weir_cockerham_fst)
    assert callable(allel2.windowed_hudson_fst)
    assert callable(allel2.windowed_patterson_fst)
    assert callable(allel2.moving_weir_cockerham_fst)
    assert callable(allel2.moving_hudson_fst)
    assert callable(allel2.moving_patterson_fst)
    assert callable(allel2.average_weir_cockerham_fst)
    assert callable(allel2.average_hudson_fst)
    assert callable(allel2.average_patterson_fst)
    # backwards compatibility
    assert callable(allel2.blockwise_weir_cockerham_fst)
    assert callable(allel2.blockwise_hudson_fst)
    assert callable(allel2.blockwise_patterson_fst)

    # allel2.stats.hw
    assert callable(allel2.heterozygosity_observed)
    assert callable(allel2.heterozygosity_expected)
    assert callable(allel2.inbreeding_coefficient)

    # allel2.stats.ld
    assert callable(allel2.rogers_huff_r)
    assert callable(allel2.rogers_huff_r_between)
    assert callable(allel2.locate_unlinked)
    assert callable(allel2.windowed_r_squared)
    assert callable(allel2.plot_pairwise_ld)

    # allel2.stats.mendel
    assert callable(allel2.mendel_errors)
    assert callable(allel2.paint_transmission)
    assert callable(allel2.phase_progeny_by_transmission)
    assert callable(allel2.phase_parents_by_transmission)
    assert callable(allel2.phase_by_transmission)

    # allel2.stats.misc
    assert callable(allel2.plot_variant_locator)
    assert callable(allel2.tabulate_state_transitions)
    assert callable(allel2.tabulate_state_blocks)

    # allel2.stats.preprocessing
    assert callable(allel2.get_scaler)
    assert callable(allel2.StandardScaler)
    assert callable(allel2.CenterScaler)
    assert callable(allel2.PattersonScaler)

    # allel2.stats.roh
    assert callable(allel2.roh_mhmm)
    assert callable(allel2.roh_poissonhmm)

    # allel2.stats.roh
    assert callable(allel2.ehh_decay)
    assert callable(allel2.voight_painting)
    assert callable(allel2.plot_voight_painting)
    assert callable(allel2.fig_voight_painting)
    assert callable(allel2.ihs)
    assert callable(allel2.xpehh)
    assert callable(allel2.nsl)
    assert callable(allel2.xpnsl)
    assert callable(allel2.haplotype_diversity)
    assert callable(allel2.moving_haplotype_diversity)
    assert callable(allel2.garud_h)
    assert callable(allel2.moving_garud_h)
    assert callable(allel2.plot_haplotype_frequencies)
    assert callable(allel2.moving_delta_tajima_d)
    assert callable(allel2.standardize)
    assert callable(allel2.standardize_by_allele_count)
    assert callable(allel2.pbs)

    # allel2.stats.sf
    assert callable(allel2.sfs)
    assert callable(allel2.sfs_folded)
    assert callable(allel2.sfs_scaled)
    assert callable(allel2.sfs_folded_scaled)
    assert callable(allel2.scale_sfs)
    assert callable(allel2.scale_sfs_folded)
    assert callable(allel2.joint_sfs)
    assert callable(allel2.joint_sfs_folded)
    assert callable(allel2.joint_sfs_scaled)
    assert callable(allel2.joint_sfs_folded_scaled)
    assert callable(allel2.scale_joint_sfs)
    assert callable(allel2.scale_joint_sfs_folded)
    assert callable(allel2.fold_sfs)
    assert callable(allel2.plot_sfs)
    assert callable(allel2.plot_sfs_folded)
    assert callable(allel2.plot_sfs_scaled)
    assert callable(allel2.plot_sfs_folded_scaled)
    assert callable(allel2.plot_joint_sfs)
    assert callable(allel2.plot_joint_sfs_folded)
    assert callable(allel2.plot_joint_sfs_scaled)
    assert callable(allel2.plot_joint_sfs_folded_scaled)

    # allel2.stats.window
    assert callable(allel2.moving_statistic)
    assert callable(allel2.moving_mean)
    assert callable(allel2.moving_std)
    assert callable(allel2.moving_midpoint)
    assert callable(allel2.index_windows)
    assert callable(allel2.position_windows)
    assert callable(allel2.window_locations)
    assert callable(allel2.windowed_count)
    assert callable(allel2.windowed_statistic)
    assert callable(allel2.per_base)
    assert callable(allel2.equally_accessible_windows)

    # allel2.util
    assert callable(allel2.hdf5_cache)
    # N.B., check this is not clobbered, see
    # https://github.com/cggh/scikit-allel/issues/163
    import allel2.util
    assert callable(allel2.util.hdf5_cache)
