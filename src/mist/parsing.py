""" parsing.py """
from datetime import datetime


def add_base_args(parser):
    ba = parser.add_argument_group("Base Args")
    ba.add_argument(
        "--debug",
        default=False,
        action="store",
        choices=["test", "test_overfit", "test_val"],
    )
    ba.add_argument("--seed", default=None, action="store", type=int)
    ba.add_argument("--save-dir", default="output", action="store")
    return ba


def add_hyperopt_args(parser):
    # Tune args
    ha = parser.add_argument_group("Hyperopt Args")
    ha.add_argument("--cpus-per-trial", default=1, type=int)
    ha.add_argument("--gpus-per-trial", default=1, type=float)
    ha.add_argument("--num-h-samples", default=50, type=int)
    ha.add_argument("--grace-period", default=60 * 15, type=int)  # 5)#60*15, type=int)
    ha.add_argument("--max-concurrent", default=10, type=int)
    ha.add_argument("--tune-checkpoint", default=None)
    ha.add_argument("--tune-save", default=False, action="store_true")

    # Overwrite default savedir
    time_name = datetime.now().strftime("%Y_%m_%d")
    save_default = f"results/{time_name}_hyperopt/"
    parser.set_defaults(save_dir=save_default)


def add_dataset_args(parser):
    da = parser.add_argument_group("Dataset Args")
    da.add_argument(
        "--magma-folder",
        action="store",
        help="Name of magma dir",
        default=None,
    )
    da.add_argument(
        "--subform-folder",
        action="store",
        help="Name of subformula dir",
        default=None,
    )
    da.add_argument(
        "--labels-file",
        action="store",
        help="Labels file tsv",
        default=None,
    )
    da.add_argument(
        "--spec-folder",
        action="store",
        help="Spectrum folder",
        default=None,
    )
    da.add_argument(
        "--split-file", help="Name of split inds file", action="store", default=None
    )
    da.add_argument(
        "--augment-data",
        default=False,
        action="store_true",
        help="If true, agument data",
    )
    da.add_argument(
        "--augment-prob",
        default=0.5,
        type=float,
        help="Prob of augmenting a peak at all",
    )
    da.add_argument(
        "--remove-weights",
        type=str,
        help="remove probability weights",
        choices=["quadratic", "uniform", "exp"],
        default="exp",
    )

    # Define arg inten-transform
    da.add_argument(
        "--inten-transform",
        default="float",
        action="store",
        choices=["float", "cat", "log", "zero"],
    )

    da.add_argument(
        "--inten-prob", default=0.1, type=float, help="Prob of rescaling a peak"
    )
    da.add_argument(
        "--remove-prob", default=0.5, type=float, help="Prob of rescaling a peak"
    )
    da.add_argument(
        "--forward-labels",
        default=None,
        action="store",
        help="Name of forward labels file",
    )
    da.add_argument(
        "--forward-aug-folder",
        default=None,
        action="store",
        help="Forward augmentation folder",
    )
    da.add_argument(
        "--frac-orig",
        default=0.4,
        type=float,
        help="Frac original data going into each batch",
    )
    return da


def add_xformer_args(parser):
    ma = parser.add_argument_group("Model args")
    ma.add_argument(
        "--loss-fn",
        default="cosine",
        action="store",
        type=str,
        help="Loss fn name",
        choices=["bce", "mse", "cosine"],
    )
    ma.add_argument(
        "--fp-file",
        default=None,
        action="store",
        help="Name of fp file with cached outputs"
    )
    ma.add_argument(
        "--fp-names",
        action="store",
        nargs="+",
        help="List of fp names for pred",
        default=["morgan2048"],
        choices=[
            "morgan512",
            "morgan1024",
            "morgan2048",
            "morgan_project",
            "morgan4096",
            "morgan4096_3",
            "maccs",
            "csi",
        ],
    )
    ma.add_argument(
        "--shuffle-train",
        default=False,
        action="store_true",
        help="If true, shuffle target order",
    )
    ma.add_argument(
        "--iterative-preds",
        default="none",
        action="store",
        choices=["none", "growing"],
        help=(
            "If not none, re-stack predictions iteratively:\n"
            " growing: Growing larger modulo fps"
        ),
    )
    ma.add_argument(
        "--iterative-loss-weight",
        default=0.5,
        type=float,
        help="Iterative loss weight for each layer",
    )
    ma.add_argument(
        "--refine-layers", default=1, type=int, help="Number of refinement layrs"
    )
    ma.add_argument("--hidden-size", type=int, help="NN Hidden size", default=50)
    ma.add_argument(
        "--num-spec-layers",
        type=int,
        help="number of spectra encoder layers",
        default=2,
    )
    ma.add_argument(
        "--spectra-dropout",
        type=float,
        default=0.1,
        help="Amount of dropout in spectra encoder",
    )
    ma.add_argument(
        "--embed-instrument",
        default=False,
        action="store_true",
        help="If true, embed the instrument",
    )
    return ma


def add_ffn_args(parser):
    ma = parser.add_argument_group("Model args")
    ma.add_argument(
        "--loss-fn",
        default="cosine",
        action="store",
        type=str,
        help="Loss fn name",
        choices=["bce", "mse", "cosine"],
    )
    ma.add_argument(
        "--fp-file",
        default=None,
        action="store",
        help="Name of fp file with cached outputs"
    )
    ma.add_argument(
        "--fp-names",
        action="store",
        nargs="+",
        help="List of fp names for pred",
        default=["morgan2048"],
        choices=[
            "morgan512",
            "morgan1024",
            "morgan2048",
            "morgan_project",
            "morgan4096",
            "morgan4096_3",
            "maccs",
            "csi",
        ],
    )
    ma.add_argument(
        "--shuffle-train",
        default=False,
        action="store_true",
        help="If true, shuffle target order",
    )
    ma.add_argument(
        "--iterative-preds",
        default="none",
        action="store",
        choices=["none", "growing"],
        help=(
            "If not none, re-stack predictions iteratively:\n"
            " growing: Growing larger modulo fps"
        ),
    )
    ma.add_argument(
        "--iterative-loss-weight",
        default=0.5,
        type=float,
        help="Iterative loss weight for each layer",
    )
    ma.add_argument(
        "--refine-layers", default=1, type=int, help="Number of refinement layrs"
    )
    ma.add_argument("--hidden-size", type=int, help="NN Hidden size", default=50)
    ma.add_argument(
        "--num-spec-layers",
        type=int,
        help="number of spectra encoder layers",
        default=2,
    )
    ma.add_argument(
        "--spectra-dropout",
        type=float,
        default=0.1,
        help="Amount of dropout in spectra encoder",
    )
    ma.add_argument(
        "--num-bins",
        action="store",
        help="Bins for binned spec",
        default=2000,
        type=int,
    )
    ma.add_argument(
        "--embed-instrument",
        default=False,
        action="store_true",
        help="If true, embed the instrument",
    )
    return ma


def add_mist_args(parser):
    ma = parser.add_argument_group("Model args")
    ma.add_argument(
        "--loss-fn",
        default="cosine",
        action="store",
        type=str,
        help="Loss fn name",
        choices=["bce", "mse", "cosine"],
    )
    ma.add_argument(
        "--no-diffs",
        default=False,
        action="store_true",
        help="If true, do not use differences at each peak",
    )
    ma.add_argument(
        "--top-layers",
        default=1,
        type=int,
        help="Number of top layers required",
    )
    ma.add_argument(
        "--fp-file",
        default=None,
        action="store",
        help="Name of fp file with cached outputs"
    )
    ma.add_argument(
        "--fp-names",
        action="store",
        nargs="+",
        help="List of fp names for pred",
        default=["morgan2048"],
        choices=[
            "morgan512",
            "morgan1024",
            "morgan2048",
            "morgan_project",
            "morgan4096",
            "morgan4096_3",
            "maccs",
            "csi",
        ],
    )
    ma.add_argument(
        "--shuffle-train",
        default=False,
        action="store_true",
        help="If true, shuffle target order",
    )
    ma.add_argument(
        "--iterative-preds",
        default="none",
        action="store",
        choices=["none", "growing"],
        help=(
            "If not none, re-stack predictions iteratively:\n"
            " growing: Growing larger modulo fps"
        ),
    )
    ma.add_argument(
        "--iterative-loss-weight",
        default=0.5,
        type=float,
        help="Iterative loss weight for each layer",
    )
    ma.add_argument(
        "--refine-layers", default=1, type=int, help="Number of refinement layrs"
    )
    ma.add_argument("--hidden-size", type=int, help="NN Hidden size", default=50)
    ma.add_argument("--max-peaks", type=int, help="Max number of peaks", default=None)
    ma.add_argument(
        "--spectra-dropout",
        type=float,
        default=0.1,
        help="Amount of dropout in spectra encoder",
    )

    ma.add_argument(
        "--magma-loss-lambda",
        default=0.5,
        type=float,
        help="Factor with which to scale the fragment fingerprint predictor loss",
    )
    ma.add_argument(
        "--magma-modulo",
        default=2048,
        type=int,
        help="How much to fold the magma fingerprint bits for prediction",
    )
    ma.add_argument(
        "--form-embedder",
        action="store",
        help="Formula embedder",
        default="float",
        choices=[
            "float",
            "abs-sines",
            "learnt",
            "fourier-sines",
            "rbf",
            "one-hot",
            "pos-cos",
        ],
    )

    ma.add_argument(
        "--magma-aux-loss", default=False, action="store_true", help="Use magma loss"
    )

    ma.add_argument(
        "--peak-attn-layers",
        type=int,
        help="Number of peak attn layers ",
        default=1,
    )
    ma.add_argument(
        "--num-heads",
        type=int,
        help="Number of attn heads",
        default=8,
    )
    ma.add_argument(
        "--pairwise-featurization",
        default=False,
        action="store_true",
        help="If true, use pairwise featurizations ontop of attention",
    )
    ma.add_argument(
        "--embed-instrument",
        default=False,
        action="store_true",
        help="If true, embed the instrument",
    )

    ma.add_argument(
        "--cls-type",
        default="ms1",
        action="store",
        choices=["ms1", "zeros"],
        help="How to store the CLS token",
    )

    ma.add_argument(
        "--set-pooling",
        type=str,
        help="Set pooling strategy",
        choices=["intensity", "root", "mean", "cls"],
        default="cls",
    )

    return ma


def add_contrastive_args(parser):
    """add_contrastive_args.

    Args:
        parser:
    """
    ma = parser.add_argument_group("Model args")
    ma.add_argument(
        "--hdf-file",
        action="store",
        type=str,
        help="Name of hdf file for retrieval",
    )
    ma.add_argument(
        "--fp-file",
        default=None,
        action="store",
        help="Name of fp file with cached outputs"
    )

    ma.add_argument(
        "--no-pretrain-load",
        help="If true, don't load pretrained model",
        default=False,
        action="store_true",
    )

    ma.add_argument(
        "--dist-name",
        default="bce",
        action="store",
        type=str,
        help="Name of distance",
        choices=["bce", "tanimoto", "euclid", "cosine", "learned"],
    )
    ma.add_argument(
        "--contrastive-loss",
        action="store",
        default="nce",  # "softmax",
        choices=["clip", "softmax", "nce", "triplet", "triplet_rand", "none"],
        help="name of contrast loss",
    )
    ma.add_argument(
        "--contrastive-decoy-pool",
        action="store",
        default="mean",
        choices=["mean", "max", "logsumexp"],
        help="How to pool contrastive triplets",
    )
    ma.add_argument(
        "--contrastive-latent",
        action="store",
        default="h0",
        choices=["fp", "h0", "aux", "fp_aux", "fp_aux_siamese", "h0_learned"],
        help=(
            "Defines in what space we should have "
            "contrastive penalty take place. Options:\n"
            "fp: Contrastive loss on fingerprints\n"
            "h0: Contrastive loss on the output latent\n"
            "aux: Contrastive loss on a sep. output\n"
            "fp_aux: Contrastive loss ontop of fp\n"
            "fp_aux_siamese: FP_aux with siamese encoders\n"
            "h0_learned: h0 but with learned dist\n"
        ),
    )
    ma.add_argument(
        "--contrastive-weight",
        action="store",
        default=1.0,
        type=float,
        help="Amt to wieght contrastive loss",
    )
    ma.add_argument(
        "--num-decoys",
        action="store",
        default=5,
        type=int,
        help="Number of decoys per example",
    )
    ma.add_argument(
        "--max-db-decoys",
        action="store",
        default=512,
        type=int,
        help="Max database decoys",
    )
    ma.add_argument(
        "--decoy-norm-exp",
        action="store",
        default=None,
        type=float,
        help="If set, use softmax to renormalize decoy weights with this scale factor",
    )
    ma.add_argument(
        "--contrastive-latent-size",
        action="store",
        default=256,
        type=int,
        help="Size of contrastive latent size",
    )
    ma.add_argument(
        "--contrastive-latent-dropout",
        action="store",
        default=0.0,
        type=float,
        help="Contrastive dropout amount",
    )
    ma.add_argument(
        "--contrastive-scale",
        action="store",
        default=1,
        type=float,
        help="Scale factor for softmax term. Helps create soft margin",
    )
    ma.add_argument(
        "--contrastive-bias",
        action="store",
        default=0,
        type=float,
        help="Bias term for contrastive loss",
    )
    ma.add_argument(
        "--negative-strategy",
        action="store",
        default="random",
        choices=[
            "random",
            "hardisomer_tani_pickled",
        ],
        help="name of contrast loss",
    )
    return ma


def add_train_args(parser):
    ta = parser.add_argument_group("Train args")
    ta.add_argument(
        "--learning-rate", help="Learning rate for model", default=1e-3, type=float
    )
    ta.add_argument(
        "--weight-decay", help="Weight decay for model", default=0, type=float
    )
    ta.add_argument(
        "--lr-decay-frac",
        help="LR decay fraction every k steps",
        default=0.995,
        type=float,
    )
    ta.add_argument(
        "--scheduler",
        help="If true, use a scheduler",
        default=False,
        action="store_true",
    )
    ta.add_argument(
        "--patience",
        help="Amt of patience to use",
        default=20,
        action="store",
        type=int,
    )
    ta.add_argument(
        "--ckpt-file",
        help="Name of ckpt file",
        default=None,
        type=str,
    )

    ta.add_argument(
        "--min-epochs",
        type=int,
        help="Min epochs",
        default=None,
    )
    ta.add_argument(
        "--gpus",
        type=int,
        help="Num gpus",
        default=0,
    )
    ta.add_argument(
        "--max-epochs",
        type=int,
        help="Max epochs",
        default=600,
    )
    ta.add_argument("--batch-size", type=int, default=32)
    ta.add_argument("--num-workers", help="Dataset workers", type=int, default=0)
    ta.add_argument(
        "--persistent-workers",
        help="If true, keep dataset instances alive",
        default=False,
        action="store_true",
    )
    # Featurizers
    ta.add_argument("--cache-featurizers", action="store_true", default=False)
    return ta
