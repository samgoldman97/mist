""" csi_to_mgf.py 

Useful to create MGF to be used as input to the various distance methods as
comparison.

"""

from pathlib import Path
from mist import utils

csi_spec_dir = Path("data/paired_spectra/csi2022/spec_files")
outfile = Path("data/paired_spectra/csi2022/csi2022.mgf")

spec_files = list(csi_spec_dir.glob("*"))

meta_spec_list = utils.chunked_parallel(spec_files, utils.parse_spectra)

out = utils.build_mgf_str(meta_spec_list)
with open(outfile, "w") as fp:
    fp.write(out)
