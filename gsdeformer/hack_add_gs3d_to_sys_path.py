import sys

from gsdeformer.paths import GS3D_FOLDER

if str(GS3D_FOLDER.absolute()) not in sys.path:
    sys.path.append(str(GS3D_FOLDER.absolute()))
