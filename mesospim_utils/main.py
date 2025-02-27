import typer

app = typer.Typer()


from rl2 import convert_ims, convert_ims_dir_mesospim_tiles, ims_dir, decon_dir
from stitch2 import stitch_and_assemble

convert_ims = app.command()(convert_ims)
convert_ims_dir_mesospim_tiles = app.command()(convert_ims_dir_mesospim_tiles)
ims_dir = app.command()(ims_dir)
decon_dir = app.command()(decon_dir)

stitch_and_assemble = app.command()(stitch_and_assemble)



if __name__ == "__main__":
    app()