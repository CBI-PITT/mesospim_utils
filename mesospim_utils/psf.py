#!/h20/home/lab/miniconda3/envs/decon/bin/python -u

import typer
import numpy as np

app = typer.Typer()

def get_psf(
    z: int = 7,
    nx: int = 7,
    dxy: float = 1,
    dz: float = 10,
    pz: float = 0.0,
    NA: float = 0.2,
    wvl: float = 0.595,
    ns: float = 1.33,
    ni: float = 1.33,
    ni0: float = 1.33,
    tg: float = 0,
    tg0: float = 0,
    ng: float = 1.515,
    ng0: float = 1.515,
    ti0: float = 100000,
    oversample_factor: int = 3,
    normalize: bool = True,
    model: str = "vectorial",
):
    """Compute microscope PSF.

    Select the PSF model using the `model` keyword argument. Can be one of:
        vectorial:  Vectorial PSF described by Aguet et al (2009).
        scalar:     Scalar PSF model described by Gibson and Lanni.
        gaussian:   Simple gaussian approximation.

    Parameters
    ----------
    z : Union[int, Sequence[float]]
        If an integer, z is interepreted as the number of z planes to calculate, and
        the point source always resides in the center of the z volume (at plane ~z//2).
        If a sequence (list, tuple, np.array), z is interpreted as a vector of Z
        positions at which the PSF is calculated (in microns, relative to
        coverslip).
        When an integer is provided, `dz` may be used to change the step size.
        If a sequence is provided, `dz` is ignored, since the sequence already implies
        absolute positions relative to the coverslip.
    nx : int
        XY size of output PSF in pixels, prefer odd numbers.
    dxy : float
        pixel size in sample space (microns)
    dz : float
        axial size in sample space (microns). Only used when `z` is an integer.
    pz : float
        point source z position above the coverslip, in microns.
    NA : float
        numerical aperture of the objective lens
    wvl : float
        emission wavelength (microns)
    ns : float
        sample refractive index
    ni : float
        immersion medium refractive index, experimental value
    ni0 : float
        immersion medium refractive index, design value
    tg : float
        coverslip thickness, experimental value (microns)
    tg0 : float
        coverslip thickness, design value (microns)
    ng : float
        coverslip refractive index, experimental value
    ng0 : float
        coverslip refractive index, design value
    ti0 : float
        working distance of the objective (microns)
    oversample_factor : int, optional
        oversampling factor to approximate pixel integration, by default 3
    normalize : bool
        Whether to normalize the max value to 1. By default, True.
    model : str
        PSF model to use.  Must be one of 'vectorial', 'scalar', 'gaussian'.
        By default 'vectorial'.

    Returns
    -------
    psf : np.ndarray
        The PSF array with dtype np.float64 and shape (len(zv), nx, nx)

    **DEFAULTS are optimized for CBI MesoSPIM

    Library from:
    https://github.com/tlambert03/PSFmodels
    """
    import psfmodels as psfm
    psf = psfm.make_psf(
        z = z,
        nx = nx,
        dxy = dxy,
        dz = dz,
        pz = pz,
        NA = NA,
        wvl = wvl,
        ns = ns,
        ni = ni,
        ni0 = ni0,
        tg = tg,
        tg0 = tg0,
        ng = ng,
        ng0 = ng0,
        ti0 = ti0,
        oversample_factor = oversample_factor,
        normalize = normalize,
        model = model,
        )
    print(psf)
    print(psf.shape)
    return psf.astype(np.float32)



###########################
## Depreciated Functions ##
###########################

def generate_psf_3d(size, scale, NA, RI, wavelength):
    """
    Generate a 3D PSF based on optical parameters.

    Args:
        size (tuple): Shape of the PSF (D, H, W).
        scale (tuple): Scale of voxels in microns (z, y, x).
        NA (float): Numerical aperture of the system.
        RI (float): Refractive index of the medium.
        wavelength (float): Wavelength of light in microns.

    Returns:
        torch.Tensor: PSF with the given parameters.
    """
    z, y, x = size
    z_scale, y_scale, x_scale = scale

    # Calculate diffraction-limited resolution
    resolution_xy = 0.61 * wavelength / NA
    resolution_z = 2 * wavelength / (NA ** 2)

    # Create coordinate grids
    z_range = np.linspace(-(z // 2) * z_scale, (z // 2) * z_scale, z)
    y_range = np.linspace(-(y // 2) * y_scale, (y // 2) * y_scale, y)
    x_range = np.linspace(-(x // 2) * x_scale, (x // 2) * x_scale, x)
    zz, yy, xx = np.meshgrid(z_range + 1, y_range + 1, x_range + 1, indexing="ij")

    # Generate Gaussian approximation of the PSF
    psf = np.exp(
        -((xx ** 2 + yy ** 2) / (2 * resolution_xy ** 2) + (zz ** 2) / (2 * resolution_z ** 2))
    )

    # Normalize PSF
    psf /= psf.sum()

    print(f'PSF SHAPE: {psf.shape}')
    print(psf)

    return psf
    # return torch.tensor(psf, dtype=torch.float32)


def generate_psf_3d_v2(size: tuple[int, int, int] = (3, 3, 3), scale: tuple[float, float, float] = (1, 1, 1),
                       na: float = 0.2, ri: float = 1.0, wavelength: int = 488):
    """
    size, scale, NA, RI, wavelength
    Estimate the 3D PSF for a microscopy system.

    Parameters:
    - na: Numerical Aperture (unitless)
    - wavelength: Emission wavelength (in nanometers)
    - resolution_xy: Pixel size in the xy-plane (in microns)
    - resolution_z: Pixel size in the z-dimension (in microns)
    - size: Tuple indicating the shape of the PSF (z, y, x)

    Returns:
    - psf: 3D PSF array
    """

    resolution_z, resolution_xy, _ = scale

    # Convert from nm to um
    wavelength /= 1000

    # Adjust na and wavelength for RI
    na = min(na, ri)
    wavelength = wavelength / ri

    # Compute FWHM in microns
    fwhm_xy = 0.61 * wavelength / na
    fwhm_z = 2 * wavelength / (na ** 2)

    # Convert FWHM to standard deviation in voxels
    sigma_xy = fwhm_xy / (2 * np.sqrt(2 * np.log(2))) / resolution_xy
    sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2))) / resolution_z

    # Create a 3D Gaussian kernel
    z, y, x = size
    z_range = np.linspace(-(z // 2), z // 2, z)
    y_range = np.linspace(-(y // 2), y // 2, y)
    x_range = np.linspace(-(x // 2), x // 2, x)
    zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing="ij")

    psf = np.exp(
        -((xx / sigma_xy) ** 2 + (yy / sigma_xy) ** 2 + (zz / sigma_z) ** 2) / 2
    )

    # psf /= psf.sum() # Normalize PSF (More accepted version)
    psf /= psf.max()  # Normalize PSF
    print(f'PSF SHAPE: {psf.shape}')
    # print(psf)
    for ii in psf[0, 0]:
        print('\n\n')
    return psf


def generate_psf_3d_v3(size: tuple[int, int, int] = (3, 3, 3), scale: tuple[float, float, float] = (1, 1, 1),
                       na: float = 0.2, ri: float = 1.0, wavelength: int = 488):
    """
    Auto adjust size of PSF based on threshold
    size, scale, NA, RI, wavelength
    Estimate the 3D PSF for a microscopy system.

    Parameters:
    - na: Numerical Aperture (unitless)
    - wavelength: Emission wavelength (in nanometers)
    - ri: refractive index
    - scale: Pixel size in the z,y,x planes
    - size: Tuple indicating the shape of the PSF (z, y, x)

    Returns:
    - psf: 3D PSF array
    """

    resolution_z, resolution_y, resolution_x = scale

    # Convert from nm to um
    wavelength /= 1000

    # Adjust na and wavelength for RI
    na = min(na, ri)
    wavelength = wavelength / ri

    # Compute FWHM in microns
    fwhm_xy = 0.61 * wavelength / na
    fwhm_z = 2 * wavelength / (na ** 2)

    # Convert FWHM to standard deviation in voxels
    sigma_y = fwhm_xy / (2 * np.sqrt(2 * np.log(2))) / resolution_y
    sigma_x = fwhm_xy / (2 * np.sqrt(2 * np.log(2))) / resolution_x
    sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2))) / resolution_z

    # Create a 3D Gaussian kernel
    z, y, x = size

    z_range = np.linspace(-(z // 2), z // 2, z)
    y_range = np.linspace(-(y // 2), y // 2, y)
    x_range = np.linspace(-(x // 2), x // 2, x)
    # print(z_range)
    # print(y_range)
    # print(x_range)
    zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing="ij")

    psf = np.exp(
        -((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2 + (zz / sigma_z) ** 2) / 2
    )

    # psf /= psf.sum() # Normalize PSF (More accepted version)
    psf /= psf.max()  # Normalize PSF
    print(f'PSF SHAPE: {psf.shape}')
    # print(psf)
    THRESHOLD = 0.0001
    while (psf[0] < THRESHOLD).all() and (psf[-1] < THRESHOLD).all():
        psf = psf[1:]
        psf = psf[:-1]
    while (psf[:, 0] < THRESHOLD).all() and (psf[:, -1] < THRESHOLD).all():
        psf = psf[:, 1:]
        psf = psf[:, :-1]
    while (psf[:, :, 0] < THRESHOLD).all() and (psf[:, :, -1] < THRESHOLD).all():
        psf = psf[:, :, 1:]
        psf = psf[:, :, :-1]

    for ii in psf:
        # print(ii)
        print(np.round(ii, 4))
        print('\n')
    print(psf.shape)
    return psf.astype(np.float32)



if __name__ == "__main__":
    app()
