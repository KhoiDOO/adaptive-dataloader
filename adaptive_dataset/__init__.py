import os
os.environ["DEEPLAKE_DOWNLOAD_PATH"] = "~/data/"

from adaptive_dataset.augment.aug import LargeAugmentation
from adaptive_dataset.augment.aug import SegAumentation
from adaptive_dataset.dlake_core.core import DlakeInterface