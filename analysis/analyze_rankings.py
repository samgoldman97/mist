import yaml
import logging
import numpy as np
import pickle
import pandas as pd
import argparse
from mist import utils

logging.basicConfig(level=logging.INFO)

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--labels", action="store")
    args.add_argument("--rankings", action="store")
    args.add_argument("--output", action="store")
    return args.parse_args()


def run_analysis(labels, rankings, output):

    df = pd.read_csv(labels, sep="\t")
    spec_to_ikey = dict(df[["spec", "inchikey"]].values)
    spec_to_smiles = dict(df[["spec", "smiles"]].values)

    with open(rankings , "rb") as fp:
        loaded_rankings = pickle.load(fp)

    # Meta data
    dataset_name = loaded_rankings.get("dataset_name", "")
    args = loaded_rankings.get("args", {})
    split_name = loaded_rankings.get("split_name", "")
    retrieval_settings = loaded_rankings.get("retrieval_settings", {})
    metadata = {"dataset_name": dataset_name, 
                "args": args, 
                "split_name": split_name, 
                "retrieval_settings": retrieval_settings}
    out_entries = []

    #dict_keys(['dataset_name', 'names', 'args', 'split_name', 'ranking',
    #           'dists', 'ikeys', 'smiles', 'retrieval_settings'])
    names = loaded_rankings['names']
    rankings = loaded_rankings['ranking']
    dists = loaded_rankings['dists']
    ikeys = loaded_rankings['ikeys']
    smiles = loaded_rankings['smiles']
    for name, ranking, dist, ikey, smile in zip(names, rankings, dists, ikeys,
                                                smiles):
        name = str(name)
        true_ikey = spec_to_ikey.get(name)
        true_smiles = spec_to_smiles.get(name)
        ind_eq = np.where(np.array(ikey).astype(str) == true_ikey)[0]
        if len(ind_eq) == 0:
            ind_found = None
        else:
            ind_ex = ind_eq[0]
            if len(ind_eq) > 1: 
                logging.debug(f"Warning: found multiple ikeys for {name} in lib")

            # Resort distances from low to high
            # Use conservative estimate where we guess the _worst case_ from
            # list returned
            true_dist = dist[ind_ex]
            max_ind = np.max(np.where(true_dist == np.sort(dist))[0])

            ind_found = max_ind + 1
        out_entry = {"name": name, 
                     "ind_found": float(ind_found) if ind_found is not None else None, 
                     "true_smiles": true_smiles}
        out_entries.append(out_entry)

    # Joint analytics
    max_ind = 1e10
    df = pd.DataFrame(out_entries)
    df.loc[pd.isna(df['ind_found']).values, 'ind_found'] = max_ind
    top_ks = [1, 5, 10, 20, 50, 100, 200, 500]
    metric_dict = {}
    for k in top_ks:
        metric_dict[k] = float(np.mean(df['ind_found'] <= k))

    full_out = {"individuals": out_entries, 
                "metrics": metric_dict,
                "metadata": metadata}
    with open(output, "w") as fp:
        yaml_str = yaml.dump(full_out, indent=2)
        fp.write(yaml_str)
    logging.info(f"Retrieval results:\n{yaml.dump(metric_dict, indent=2)}")


if __name__=="__main__":
    args = get_args()
    run_analysis(**args.__dict__)


