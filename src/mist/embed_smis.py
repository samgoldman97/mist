""" embed_smis.py

Embed smiles strings using a contrastive model

"""
import copy
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm

from mist.models import base
from mist.data import featurizers, data
from mist import utils


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True, help="Model ckpt to load from")
    parser.add_argument(
        "--smiles-list", required=True, help="File csv containing a column for smiles"
    )
    parser.add_argument("--save-dir", required=False, default=None, help="Save dir")
    parser.add_argument(
        "--num-workers", action="store", type=int, help="Get num workers", default=16
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    return parser.parse_args()


def embed_smis():
    """Embed smiles strings"""
    args = get_args()
    kwargs = args.__dict__
    debug = kwargs.get("debug")
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    max_count = 100 if debug else None
    kwargs["max_count"] = max_count

    # Load saved model
    model_ckpt = kwargs.get("model_ckpt")
    pretrain_ckpt = torch.load(model_ckpt, map_location=torch.device("cpu"))
    main_hparams = pretrain_ckpt["hyper_parameters"]

    # Set save dir and setup model
    if kwargs.get("save_dir") is None:
        save_dir = Path(main_hparams["save_dir"]) / "embed"
        kwargs["save_dir"] = save_dir
    else:
        save_dir = kwargs["save_dir"]

    # Update main hparams with kwargs and switch to only using kwargs
    base_params = main_hparams["base_model_hparams"]
    new_kwargs = copy.deepcopy(main_hparams)
    new_kwargs.update(base_params)
    new_kwargs.update(kwargs)
    kwargs = new_kwargs

    utils.setup_logger(
        save_dir, log_name=f"embed_contrast.log", debug=kwargs.get("debug", False)
    )

    # Construct model and load in state dict
    model = base.build_model(**main_hparams)
    logging.info(f"Loading from epoch {pretrain_ckpt['epoch']}")
    model.load_state_dict(pretrain_ckpt["state_dict"])
    model = model.to(device)
    model = model.eval()

    # Add spec feaatures (no mol features)
    kwargs["spec_features"] = model.main_model.spec_features(mode="test")
    kwargs["mol_features"] = model.main_model.mol_features(mode="test")

    # Get mol featurizer
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    # Get embed list
    smi_list = pd.read_csv(args.smiles_list, sep="\t")
    mol_list = [data.Mol.MolFromSmiles(i) for i in smi_list["smiles"].values]
    fp_list = [paired_featurizer.featurize_mol(i) for i in mol_list]

    # Encode all hdf fp's with single model
    fp_list = torch.from_numpy(np.vstack(fp_list))

    # Get the HDF entries for each of these
    fp_loader = torch.utils.data.DataLoader(fp_list, batch_size=128, shuffle=False)
    encoded_mols = []
    logging.info("Encoding mols")
    with torch.no_grad():
        model = model.to(device)
        model = model.eval()
        for fp_batch in tqdm(fp_loader):
            batch = {"mols": fp_batch.to(device)}

            outputs = model.encode_mol(batch)
            contrast_out = outputs[1]["contrast"].detach().cpu()
            encoded_mols.append(contrast_out)
        stacked_mols = torch.cat(encoded_mols, 0).numpy()

    # Map inchikey to output
    smis = smi_list["smiles"].values
    ikeys = smi_list["inchikey"].values
    names = smi_list["name"].values
    output = {"smiles": smis, "inchikey": ikeys, "names": names, "embeds": stacked_mols}

    # Dump to output file
    ctr = 0
    f_name = Path(save_dir) / f"embed_smiles_{ctr}.p"
    while (f_name).exists():
        ctr += 1
        f_name = Path(save_dir) / f"embed_smiles_{ctr}.p"

    # Output is new_entry
    with open(f_name, "wb") as fp:
        pickle.dump(output, fp)


if __name__ == "__main__":
    import time

    start_time = time.time()
    embed_smis()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
