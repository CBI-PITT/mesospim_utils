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
    '300-479': (0, 0, 0.5),
    '480-540': (0, 0.5, 0),
    '541-625': (0.5, 0, 0),
    '627-730': (0.75, 0, 0.5),
    '731-2000': (0.5, 0, 0.75),
}