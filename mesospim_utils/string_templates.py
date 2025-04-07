
from constants import PATH_TO_IMARIS_STITCHER_FOLDER

from constants import FRACTION_OF_RAM_FOR_PROCESSING
from constants import IMS_CONVERTER_COMPRESSION_LEVEL
from utils import get_ram_mb

memlimit_for_processing = int(get_ram_mb() * FRACTION_OF_RAM_FOR_PROCESSING)

WIN_ALIGN_BAT = f'''@echo off
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER};%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\jre;%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\jre\\bin;%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\jre\\bin\server;%PATH%"
::echo %PATH%

cd /d "{PATH_TO_IMARIS_STITCHER_FOLDER}"
echo Working directory set to: %CD%

"{PATH_TO_IMARIS_STITCHER_FOLDER}\ImarisStitchAlignPairwise.exe"^
 -e "{'{}'}"^
 -o "{'{}'}"^
 -s 4.243^
 -c {'{}'}^
 -b true^
 --jobs 64^
 --memlimit {memlimit_for_processing}'''

WIN_RESAMPLE_BAT = f'''@echo off
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER};%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\jre;%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\jre\\bin;%PATH%"
set "PATH={PATH_TO_IMARIS_STITCHER_FOLDER}\jre\\bin\server;%PATH%"
::echo %PATH%

cd /d "{PATH_TO_IMARIS_STITCHER_FOLDER}"
echo Working directory set to: %CD%

"{PATH_TO_IMARIS_STITCHER_FOLDER}\ImarisStitchResample.exe"^
 -e "{'{}'}"^
 -o "{'{}'}"^
 --fuseRule weighted-mean^
 --color "{'{}'}"^
 --jobs {'{}'}^
 --memlimit {memlimit_for_processing}^
 --compression eCompressionAlgorithmGzipLevel{IMS_CONVERTER_COMPRESSION_LEVEL}'''


COLOR_RECORD_TEMPLATE = '''<Channel ChannelIndex="Channel {}" Selection="true" RangeMax="{}" RangeMin="{}" GammaCorrection="1" Opacity="1" ColorMode="BaseColor" RangeMinB="3.40282e+38" RangeMaxB="3.40282e+38">
<ColorTable/>
<BaseColor>
<Color Red="{}" Green="{}" Blue="{}"/>
</BaseColor>
</Channel>'''

# BAT_RENAME_FILE = f'\n\nif [ -f "{out_file}.part" ]; then\n  mv "{out_file}.part" "{out_file}"\n  echo "File renamed to {out_file}"\nelse\n  echo "File {out_file} does not exist."\nfi'
