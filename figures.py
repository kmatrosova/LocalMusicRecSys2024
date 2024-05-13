from helpers import MetadataProcesor, MakePlots


N_TRIES = "max"
K_VALUES = range(10, 101, 5)
LFM_metadata_file_name = "metadata_LFM"
DEEZER_metadata_file_name_list = [
    "metadata_DEEZER_active",
    "metadata_DEEZER_musicbrainz",
    "metadata_DEEZER_origin",
]

for GLOBAL_VALUE in [True, False]:
    for DEEZER_metadata_file_name in DEEZER_metadata_file_name_list:

        metadata = MetadataProcesor(LFM_metadata_file_name, DEEZER_metadata_file_name)
        metadata.process()

        plots_maker = MakePlots(
            metadata=metadata.metadata,
            k_values=K_VALUES,
            matadata_filename=DEEZER_metadata_file_name,
            global_models=GLOBAL_VALUE,
        )

        plots_maker.load_predictions(n_tries=N_TRIES)
        plots_maker.load_datasets()

        plots_maker.plot_dataset_local_streams_percents(save=True)
        plots_maker.plot_local_listening_distribution_hist(save=True)

        plots_maker.compute_reco_results()
        plots_maker.plot_bias_topk_k_reco(save=True)
