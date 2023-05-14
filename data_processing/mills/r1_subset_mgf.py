from pathlib import Path
from mist import utils


mass_thresh = 500
start_path = Path("data/raw/mills/Mills_mzxml/mgf_export_sirius.mgf")
out_path = Path(f"data/raw/mills/Mills_mzxml/mgf_export_sirius_filtered_{mass_thresh}.mgf")

parsed_spec = utils.parse_spectra_mgf(start_path)
new_spec = [(i[0], i[1]) for i in parsed_spec if float(i[0]['PEPMASS']) <= mass_thresh]
out_str = utils.build_mgf_str(new_spec)

with open(out_path, "w") as fp:
    fp.write(out_str)
