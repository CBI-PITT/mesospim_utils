
from constants import PATH_TO_IMARIS_STITCHER_FOLDER, PATH_TO_IMARIS_STITCHER_TEMP_FOLDER, NUM_CPUS_FOR_STITCH

from constants import FRACTION_OF_RAM_FOR_PROCESSING
from constants import IMS_CONVERTER_COMPRESSION_LEVEL
from utils import get_ram_mb

memlimit_for_processing = int(get_ram_mb() * FRACTION_OF_RAM_FOR_PROCESSING)


CHANGE_WINDOWS_TMP_FOLDER = f''':: Base temp parent folder
set BASE_TEMP_DIR={PATH_TO_IMARIS_STITCHER_TEMP_FOLDER}

:: Create a random subfolder name
set TEMP_SUBDIR=tmp_stitch_%RANDOM%_%RANDOM%
set TEMP=%BASE_TEMP_DIR%\\%TEMP_SUBDIR%
set TMP=%TEMP%

:: Create the directory
mkdir "%TEMP%"

echo Using temporary folder: %TEMP%'''



CLEAN_UP_TEMP_FILES = ''':: --- Cleanup Temp Files ---
echo Cleaning up temp files...

if exist "%TEMP%" (
    echo Deleting TEMP folder: %TEMP%
    :: Remove the temp directory and all contained data
    rmdir /s /q "%TEMP%"
) else (
    echo TEMP folder not found: %TEMP%
)'''



WIN_ALIGN_BAT = f'''@echo off
setlocal

{CHANGE_WINDOWS_TMP_FOLDER if PATH_TO_IMARIS_STITCHER_TEMP_FOLDER is not None else ""}

set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER};%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\\jre;%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\\jre\\bin;%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\\jre\\bin\\server;%PATH%"
::echo %PATH%

cd /d "{PATH_TO_IMARIS_STITCHER_FOLDER}"
echo Working directory set to: %CD%

"{PATH_TO_IMARIS_STITCHER_FOLDER}\\ImarisStitchAlignPairwise.exe"^
 -e "{'{}'}"^
 -o "{'{}'}"^
 -s 4.243^
 -c {'{}'}^
 -b true^
 --jobs {NUM_CPUS_FOR_STITCH}^
 --memlimit {memlimit_for_processing}
 
{CLEAN_UP_TEMP_FILES if PATH_TO_IMARIS_STITCHER_TEMP_FOLDER is not None else ""}'''



WIN_RESAMPLE_BAT = f'''@echo off
setlocal

{CHANGE_WINDOWS_TMP_FOLDER if PATH_TO_IMARIS_STITCHER_TEMP_FOLDER is not None else ""}

set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER};%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\\jre;%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\\jre\\bin;%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\\jre\\bin\\server;%PATH%"
::echo %PATH%

cd /d "{PATH_TO_IMARIS_STITCHER_FOLDER}"
echo Working directory set to: %CD%

"{PATH_TO_IMARIS_STITCHER_FOLDER}\\ImarisStitchResample.exe"^
 -e "{'{}'}"^
 -o "{'{}'}"^
 --fuseRule weighted-mean^
 --color "{'{}'}"^
 --jobs {NUM_CPUS_FOR_STITCH}^
 --memlimit {memlimit_for_processing}^
 --compression eCompressionAlgorithmGzipLevel{IMS_CONVERTER_COMPRESSION_LEVEL}

{CLEAN_UP_TEMP_FILES if PATH_TO_IMARIS_STITCHER_TEMP_FOLDER is not None else ""}'''


COLOR_RECORD_TEMPLATE = '''<Channel ChannelIndex="Channel {}" Selection="true" RangeMax="{}" RangeMin="{}" GammaCorrection="1" Opacity="1" ColorMode="BaseColor" RangeMinB="3.40282e+38" RangeMaxB="3.40282e+38">
<ColorTable/>
<BaseColor>
<Color Red="{}" Green="{}" Blue="{}"/>
</BaseColor>
</Channel>'''
