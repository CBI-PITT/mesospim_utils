

## Change this path for any specific
PATH_TO_IMARIS_STITCHER_FOLDER = r"C:\Program Files\Bitplane\ImarisStitcher 10.2.0"
SHARED_WINDOWS_PATH_WHERE_WIN_CLIENT_JOB_FILES_ARE_STORED = r"Z:\tmp\stitch_jobs"

METADATA_FILENAME = 'mesospim_metadata.json'

CORRELATION_THRESHOLD_FOR_ALIGNMENT = 0.5

PERCENT_OF_RAM_FOR_PROCESSING = 0.5

EMISSION_MAP = {
    #Mapping common names to emission wavelengths, only used if wavelength is not explicitly stated in metadata file
    "gfp": 525,
    "green": 525,
    "rfp": 595,
    "red": 595,
    "cy5": 665,
    "far_red": 665,
}

EMISSION_TO_RGB = {
    #Mapping emission range to RGB colors
    # Keys are wavelength ranges in nm
    # Values are (R,G,B)
    '300-479': (0, 0, 1),
    '480-540': (0, 1, 0),
    '541-625': (1, 0, 0),
    '627-730': (1, 0, 0.75),
    '731-2000': (.75, 0, 1),
}