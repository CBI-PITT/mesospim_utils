#!/h20/home/lab/miniconda3/envs/mesospim_mv_dev/bin/python -i

if __name__ == "__main__":



    import sys

    ## Add paths so imports work
    sys.path.insert(
        0,
        "/CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils"
    )

    from pathlib import Path
    from metadata import collect_all_metadata, get_first_entry, get_each_tile
    from utils import ensure_path

    # from multiview_stitcher import spatial_image_utils as si_utils
    from multiview_stitcher import (
        registration,
    #     fusion,
    #     param_utils,
    #     msi_utils,
    #     misc_utils,
    #     vis_utils,
    #     ngff_utils,
    )
    from multiview_stitcher.ngff_utils import read_msim_from_ome_zarr


    import xarray as xr

    dataset_location = (
        "/CBI_FastStore/test_data/mesospim/omezarr/072126_green_embryo_3channels_3x3tiles"
    )

    # dataset_location = (
    #     "/CBI_FastStore/test_data/mesospim/omezarr/"
    #     "072126_green_embryo_3channels_3x3tiles"
    # )

    dataset_location = ensure_path(dataset_location)
    metadata_by_channel = collect_all_metadata(dataset_location)

    print(dataset_location)
    first_entry = get_first_entry(metadata_by_channel)

    tile_file_name = first_entry.get('file_path')
    print(tile_file_name)

    def set_channel_label(ds: xr.Dataset, channel_label: str) -> xr.Dataset:
        if "c" not in ds.dims:
            return ds

        if ds.sizes["c"] != 1:
            raise ValueError(
                f"Expected one channel, but dataset contains "
                f"{ds.sizes['c']} channels"
            )

        return ds.assign_coords(
            c=("c", [channel_label])
        )

    msims = {}
    for channel in metadata_by_channel:
        if channel not in msims:
            msims[channel] = []
        for tile in metadata_by_channel[channel]:
            print(f'Reading data for: {tile.get('file_path')}')
            msim = read_msim_from_ome_zarr(
                tile.get('file_path'),
                transform_key='ome-zarr'
                )
            channel_label = tile.get('channel_label')
            msim = msim.map_over_datasets(
                set_channel_label,
                channel_label,
            )
            msims[channel].append(msim)


    print(msims[channel][-1])

    # # visualize the tile configuration and check it's properly set
    # # from multiview_stitcher import vis_utils, msi_utils, fusion
    #
    # headless = True
    # if headless:
    #     import matplotlib
    #     matplotlib.use("Agg")
    #     import matplotlib.pyplot as plt
    # from multiview_stitcher import vis_utils
    #
    # print(f'Visualizing Channels: {list(metadata_by_channel.keys())}')
    # for channel in metadata_by_channel:
    #
    #     output_path = dataset_location / Path(f'output_{channel}.png')
    #
    #     print(f'Visualizing channel: {channel}')
    #     vis_utils.plot_positions(
    #         msims[channel], transform_key='ome-zarr'
    #     )
    #
    #     if headless:
    #         figure = plt.gcf()
    #
    #         figure.suptitle(
    #             f"Tile Positions — {channel}",
    #             fontsize=16,
    #         )
    #
    #         figure.savefig(
    #             output_path,
    #             dpi=300,
    #             bbox_inches="tight",
    #         )
    #
    #         plt.close(figure)
    #
    #         print(f"Saved tile-position plot to: {output_path}")


    #####
    ## ALIGN
    #####

    from dask.delayed import delayed
    from dask import compute
    import dask.diagnostics

    # select a resolution level for registration
    reg_res_level = 3

    for channel in metadata_by_channel:
        reg_channel = channel
        break

    print(f'Registering with channel: {reg_channel}')
    with dask.diagnostics.ProgressBar():
        registration.register(
                msims[reg_channel],
                transform_key='ome-zarr',
                new_transform_key='phase_corr_registered',
                reg_channel_index=0,
                # registration_binning={'z': 2, 'y': 2, 'x': 2},
                pre_registration_pruning_method='keep_axis_aligned',
                reg_res_level=reg_res_level,
                n_parallel_pairwise_regs=10, # trade-off speed vs memory requirements (estimate of required memory: 2 * n_parallel_pairwise_regs * overlap_data_size))
                plot_summary=True,
            )



