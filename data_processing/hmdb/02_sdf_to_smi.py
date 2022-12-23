""" 02_sdf_to_smi.

Process hmdb sdf file and convert to smiles.

"""
import re
import pickle
from itertools import groupby

from tqdm import tqdm

import mist.utils as utils


def process_sdf_line(line: list):
    """process_sdf_line.

    Args:
        line (Iterator): line
    Return smi, database id
    """

    NAME_STRING = r"<(.*)>"
    # mol_entry = "".join(line)
    entry_iterator = groupby(line, key=lambda x: x.startswith("> "))
    output_dict = {}

    # Advance past the first one, since this is a mol block
    next(entry_iterator)
    for new_field, field in entry_iterator:
        name = "".join(field).strip()

        data = "".join(next(entry_iterator)[1]).strip()
        name = re.findall(NAME_STRING, name)
        if len(name) == 0:
            continue
        name = name[0]

        # Process
        if name == "DATABASE_ID":
            output_dict[name] = data
        else:
            output_dict[name] = data
    return (output_dict.get("SMILES"), output_dict.get("DATABASE_ID"))


def get_hmdb_smi_list(hmdb_file, debug=False):
    """get_hmdb_smi_list.

    Return:
        All smiles in hmdb

    """
    smi_list = []

    # Get each separate sdf entry
    key_func = lambda x: "$$" in x
    lines_to_process = []
    with open(hmdb_file, "r") as fp:
        for index, (is_true, line) in tqdm(enumerate(groupby(fp, key=key_func))):
            if is_true:
                pass
            else:
                lines_to_process.append(list(line))
            if debug and index > 20:
                break
    # Process each sdf line individually
    smi_list = [process_sdf_line(i) for i in lines_to_process]
    return smi_list


if __name__ == "__main__":
    hmdb_in_file = "data/raw/hmdb/structures.sdf"
    hmdb_out_file = "data/raw/hmdb/smiles.txt"
    hmdb_map_file = "data/raw/hmdb/hmdb_ikey_to_id.p"

    smi_list, id_list = zip(*get_hmdb_smi_list(hmdb_in_file, debug=False))

    smi_list = utils.chunked_parallel(smi_list, utils.achiral_smi)
    ikeys = utils.chunked_parallel(smi_list, utils.inchikey_from_smiles)
    ikey_to_id = dict(zip(ikeys, id_list))

    out_txt = "\n".join(smi_list)
    with open(hmdb_out_file, "w") as fp:
        fp.write(out_txt)

    with open(hmdb_map_file, "wb") as f:
        pickle.dump(ikey_to_id, f)
