import argparse
import json
import os
import os.path as osp


def save_args(args, path):
    """保存args参数到文件"""
    args_dict = vars(args)  # 将Namespace对象转换为字典

    with open(path, "w") as f:
        json.dump(
            args_dict, f, indent=4, default=str
        )  # default=str处理无法序列化的对象


def load_args(path):
    with open(path, "r") as f:
        args_dict = json.load(f)
    return args_dict


def parser_add_main_args(parser):
    parser.add_argument(
        "--detection_model",
        type=str,
        default="DualChanEnergy",
    )
    parser.add_argument("--dataset", type=str, default="amazon-ratings")
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "Datasets", "NodeData")
    parser.add_argument("--data_dir", type=str, default=path)
    parser.add_argument(
        "--ood_type",
        type=str,
        default="label",
        choices=["structure", "label", "feature"],
    )
    parser.add_argument("--threshold_division", type=int, default=5)  # energy division
    parser.add_argument("--ood_year", type=int, default=2020)  # for ogbn-arxiv
    parser.add_argument(
        "--ind_data",
        type=str,
    )
    parser.add_argument(
        "--ood_data",
        type=str,
    )
    parser.add_argument("--T", type=float, default=1.0, help="temperature for Softmax")
    parser.add_argument(
        "--train_prop", type=float, default=0.8, help="training label proportion"
    )
    parser.add_argument(
        "--valid_prop", type=float, default=0.1, help="validation label proportion"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3, help="number of distinct runs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--backbone", type=str, default="gcn")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--use_UB", type=int, default=0)
    parser.add_argument(
        "--use_reg", type=int, default=0, help="energy regularization loss"
    )
    parser.add_argument(
        "--lamda", type=float, default=1.0, help="weight for regularization"
    )
    parser.add_argument(
        "--m_in", type=float, default=-5, help="upper bound for in-distribution energy"
    )
    parser.add_argument(
        "--m_out", type=float, default=-1, help="lower bound for in-distribution energy"
    )

    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--use_bn", type=int, default=0)

    parser.add_argument("--knn_k", type=int, default=20)

    parser.add_argument("--use_prop", type=int, default=1)
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of layers for energy belief propagation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="weight for residual connection in propagation",
    )
    parser.add_argument("--energy_attention", type=int, default=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    args = parser.parse_args()

    path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "configs",
    )
    os.makedirs(path, exist_ok=True)
    save_args(args, osp.join(f"{path}", f"{args.dataset.lower()}.json"))
