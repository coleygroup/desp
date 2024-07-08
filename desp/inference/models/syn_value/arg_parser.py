"""
Adapted from ASKCOSv2 template relevance module:
https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance/-/blob/main/templ_rel_parser.py?ref_type=heads
"""


def add_model_opts(parser):
    """Model options"""
    group = parser.add_argument_group("synthetic distnce")
    group.add_argument(
        "--data_name", help="Data name", type=str, default="Synthetic Distance"
    )
    group.add_argument(
        "--model_name", help="Model name", type=str, default="Synthetic Distance"
    )
    group.add_argument("--model_type", help="Model type", type=str, default="dist")
    group.add_argument("--input_type", help="Input type", type=str, default="concat")
    group.add_argument("--max_label", help="Maximum label value", type=int, default=9)
    group.add_argument("--seed", help="random seed", type=int, default=0)
    group.add_argument(
        "--load_from", help="Trained checkpoint to load from", type=str, default=""
    )
    group.add_argument("--log_file", help="Log file name", type=str, default="default")
    group.add_argument(
        "--model_path", help="Checkpoint folder", type=str, default="./checkpoints"
    )
    group.add_argument(
        "--local_rank", help="Local process rank", type=int, default=-1, metavar="N"
    )
    group.add_argument(
        "--num_cores", help="number of cpu cores to use", type=int, default=4
    )
    group.add_argument(
        "--min_freq",
        help="minimum frequency of template in training " "data to be retained",
        type=int,
        default=1,
    )
    group.add_argument(
        "--n_templates", help="number of training templates", type=int, default=4
    )
    group.add_argument("--radius", help="fingerprint radius", type=int, default=2)
    group.add_argument("--fp_size", help="fingerprint size", type=int, default=2048)
    group.add_argument(
        "--hidden_sizes",
        help="hidden sizes, as a single comma-separated string",
        type=str,
        default="1024",
    )
    group.add_argument(
        "--hidden_activation", help="hidden activation", type=str, default="relu"
    )
    group.add_argument(
        "--skip_connection",
        help="type of skip connection " "(for backward compatibility)",
        type=str,
        choices=["none", "highway"],
        default="none",
    )
    group.add_argument(
        "--gating_activation",
        help="activation for highway gating",
        type=str,
        default="sigmoid",
    )
    group.add_argument("--dropout", help="hidden dropout", type=float, default=0.1)

    group = parser.add_argument_group("Paths")
    group.add_argument(
        "--processed_data_path",
        help="Path for saving preprocessed outputs",
        type=str,
        default="",
    )
    group.add_argument(
        "--test_output_path", help="Path for saving test outputs", type=str, default=""
    )


def add_preprocess_opts(parser):
    group = parser.add_argument_group("Preprocessing options")
    # data paths
    group.add_argument(
        "--all_reaction_file", help="All reaction file", type=str, default=""
    )
    group.add_argument(
        "--split_ratio", help="Split ratio of dataset", type=str, default="8:1:1"
    )

    group.add_argument("--train_file", help="Train file", type=str, default="")
    group.add_argument("--val_file", help="Validation file", type=str, default="")
    group.add_argument("--test_file", help="Test file", type=str, default="")


def add_train_opts(parser):
    """Training options"""
    group = parser.add_argument_group("template_relevance_train")
    group.add_argument("--backend", help="Backend for DDP", type=str, default="gloo")
    group.add_argument("--batch_size", help="Batch size", type=int, default=128)
    group.add_argument(
        "--train_batch_size", help="training batch size", type=int, default=128
    )
    group.add_argument(
        "--val_batch_size", help="validation batch size", type=int, default=256
    )
    group.add_argument(
        "--learning_rate", help="learning rate", type=float, default=1e-3
    )
    group.add_argument(
        "--clip_norm", help="Max norm for gradient clipping", type=float, default=20.0
    )
    group.add_argument("--epochs", help="num. of epochs", type=int, default=30)
    group.add_argument(
        "--early_stop", help="whether to use early stopping", action="store_true"
    )
    group.add_argument(
        "--early_stop_patience",
        help="num. of epochs w/o improvement before early stop",
        type=int,
        default=2,
    )
    group.add_argument(
        "--early_stop_min_delta",
        help="min. improvement in criteria needed to not early stop",
        type=float,
        default=1e-4,
    )
    group.add_argument(
        "--lr_scheduler_factor",
        help="factor by which to reduce LR (ReduceLROnPlateau)",
        type=float,
        default=0.3,
    )
    group.add_argument(
        "--lr_scheduler_patience",
        help="num. of epochs w/o improvement before " "reducing LR (ReduceLROnPlateau)",
        type=int,
        default=1,
    )
    group.add_argument(
        "--lr_cooldown",
        help="epochs to wait before resuming " "normal operation (ReduceLROnPlateau)",
        type=int,
        default=0,
    )


def add_predict_opts(parser):
    """Predicting options"""
    group = parser.add_argument_group("template_relevance_predict")
    group.add_argument(
        "--test_batch_size", help="Testing batch size", type=int, default=256
    )
    group.add_argument("--topk", help="Topk", type=int, default=50)
