import typer

app = typer.Typer()


from rl2 import convert_ims, convert_ims_dir_mesospim_tiles, ims_dir, decon_dir
from stitch2 import stitch_and_assemble, run_windows_auto_stitch_client, write_auto_stitch_message

convert_ims = app.command()(convert_ims)
convert_ims_dir_mesospim_tiles = app.command()(convert_ims_dir_mesospim_tiles)
ims_dir = app.command()(ims_dir)
decon_dir = app.command()(decon_dir)

stitch_and_assemble = app.command()(stitch_and_assemble)
run_windows_auto_stitch_client = app.command()(run_windows_auto_stitch_client)
write_auto_stitch_message = app.command()(write_auto_stitch_message)


if __name__ == "__main__":
    app()