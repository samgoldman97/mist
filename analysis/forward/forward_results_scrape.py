""" grab_results_brief.py """

from pathlib import Path
import pandas as pd
import yaml


in_dir = "results/2022_08_16_forward_bce_csi_fp_growth_diff_schemes"

in_dir = Path(in_dir)
keep_args = ["growing", "growing_weight", "growing_layers"]

outputs = []
for i in in_dir.rglob("test_results.yaml"):
    input_res = yaml.safe_load(open(i, "r"))
    base_entry = {j: k for j, k in input_res["args"].items() if j in keep_args}

    base_entry["test_loss"] = input_res["test_metrics"]["test_loss"]
    outputs.append(base_entry)
out_df = pd.DataFrame(outputs)
out_df.set_index(keep_args, inplace=True)
out_df = out_df.sort_index()
print(out_df)
