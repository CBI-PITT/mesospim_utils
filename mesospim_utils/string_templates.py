
from constants import PATH_TO_IMARIS_STITCHER_FOLDER

from constants import PERCENT_OF_RAM_FOR_PROCESSING
from utils import get_ram_mb

memlimit_for_processing = int(get_ram_mb() * PERCENT_OF_RAM_FOR_PROCESSING)

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
 --compression eCompressionAlgorithmGzipLevel2'''

# WIN_ALIGN_BAT = '''@echo off
# set "PATH=C:\Program Files\Bitplane\ImarisStitcher 10.2.0;%PATH%"
# set "PATH=C:\Program Files\Bitplane\ImarisStitcher 10.2.0\jre;%PATH%"
# set "PATH=C:\Program Files\Bitplane\ImarisStitcher 10.2.0\jre\\bin;%PATH%"
# set "PATH=C:\Program Files\Bitplane\ImarisStitcher 10.2.0\jre\\bin\server;%PATH%"
# ::echo %PATH%
#
# cd /d "C:\Program Files\Bitplane\ImarisStitcher 10.2.0"
# echo Working directory set to: %CD%
#
# "C:\Program Files\Bitplane\ImarisStitcher 10.2.0\ImarisStitchAlignPairwise.exe"^
#  -e "{}"^
#  -o "{}"^
#  -s 4.243^
#  -c 0^
#  -b true^
#  --jobs 64^
#  --memlimit 262064'''

# WIN_RESAMPLE_BAT = '''@echo off
# set "PATH=C:\Program Files\Bitplane\ImarisStitcher 10.2.0;%PATH%"
# set "PATH=C:\Program Files\Bitplane\ImarisStitcher 10.2.0\jre;%PATH%"
# set "PATH=C:\Program Files\Bitplane\ImarisStitcher 10.2.0\jre\\bin;%PATH%"
# set "PATH=C:\Program Files\Bitplane\ImarisStitcher 10.2.0\jre\\bin\server;%PATH%"
# ::echo %PATH%
#
# cd /d "C:\Program Files\Bitplane\ImarisStitcher 10.2.0"
# echo Working directory set to: %CD%
#
# "C:\Program Files\Bitplane\ImarisStitcher 10.2.0\ImarisStitchResample.exe"^
#  -e "{}"^
#  -o "{}"^
#  --fuseRule weighted-mean^
#  --color "{}"^
#  --jobs {}^
#  --memlimit {}^
#  --compression eCompressionAlgorithmGzipLevel2'''


COLOR_RECORD_TEMPLATE = '''<Channel ChannelIndex="Channel {}" Selection="true" RangeMax="{}" RangeMin="{}" GammaCorrection="1" Opacity="1" ColorMode="BaseColor" RangeMinB="3.40282e+38" RangeMaxB="3.40282e+38">
<ColorTable/>
<BaseColor>
<Color Red="{}" Green="{}" Blue="{}"/>
</BaseColor>
</Channel>'''

