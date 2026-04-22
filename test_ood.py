import argparse
import glob
import os.path as osp

import numpy as np
import torch
from torch_geometric import seed_everything

from args import load_args, parser_add_main_args
from logger import Logger_detect
from model import EPGNN, GRASP, DualChanEnergy, GraphEncoder, GraphSafe
from utils import eval_acc, evaluate_detect, load_dataset


def main(dataset: str, LOAD_ARGS: bool = True, SAVE_INFO: bool = True):
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    if LOAD_ARGS:
        # load config args
        args_path = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "configs",
        )
        args_dict = load_args(osp.join(f"{args_path}", f"{dataset.lower()}.json"))
        args_list = []
        for key, value in args_dict.items():
            if value is not None:
                args_list.extend([f"--{key}", str(value)])
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    seed_everything(42)
    torch.backends.cudnn.deterministic = True

    dataset_ind, _, dataset_ood_te = load_dataset(args)

    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if isinstance(dataset_ood_te, list):
        for data in dataset_ood_te:
            if len(data.y.shape) == 1:
                data.y = data.y.unsqueeze(1)
    else:
        if len(dataset_ood_te.y.shape) == 1:
            dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

    in_dim = dataset_ind.x.shape[1]
    out_dim = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])

    # load saved model
    model = GraphEncoder(in_dim, out_dim, args).to(device)
    criterion = torch.nn.NLLLoss()
    eval_func = eval_acc

    logger = Logger_detect(args.runs, args)
    model_dir = osp.join("checkpoints", f"{args.dataset.lower()}", f"{args.backbone}")

    final_result = []
    for run in range(args.runs):
        if args.dataset not in ["arxiv", "web-kb"]:
            checkpoint = glob.glob(f"{model_dir}_{args.ood_type}/model{run}*.pt")[0]
        else:
            checkpoint = glob.glob(f"{model_dir}/model{run}*.pt")[0]
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )

        # detect model
        if args.detection_model == "DualChanEnergy":
            ood_model = DualChanEnergy(model)
        elif args.detection_model == "GraphSafe":
            ood_model = GraphSafe(model)
        elif args.detection_model == "EPGNN":
            ood_model = EPGNN(model)
        elif args.detection_model == "GRASP":
            ood_model = GRASP(model)
        else:
            raise NotImplementedError

        result = evaluate_detect(
            ood_model,
            dataset_ind,
            dataset_ood_te,
            criterion,
            eval_func,
            args,
            device,
        )
        logger.add_result(run, result)
        final_result.append([result[0], result[1], result[2], result[-2]])
        print(
            f"Run: {run}, "
            f"AUROC: {100 * result[0]:.2f}, "
            f"AUPR: {100 * result[1]:.2f}, "
            f"FPR95: {100 * result[2]:.2f}, "
            f"Test Score: {100 * result[-2]:.2f}"
        )
    final_result = np.array(final_result)
    final_result_mean = np.mean(final_result, axis=0)
    final_result_std = np.std(final_result, axis=0)
    output_info = (
        f"AUROC: {100 * final_result_mean[0]:.2f}±{100 * final_result_std[0]:.2f}, "
        + f"AUPR: {100 * final_result_mean[1]:.2f}±{100 * final_result_std[1]:.2f}, "
        + f"FPR: {100 * final_result_mean[2]:.2f}±{100 * final_result_std[2]:.2f}, "
        + f"Test Score: {100 * final_result_mean[3]:.2f}±{100 * final_result_std[3]:.2f}%"
    )
    print("[Avg] " + output_info)

    # save info
    if SAVE_INFO:
        filepath = osp.join("results", f"{args.dataset}.txt")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"KNN K:{args.knn_k}")
            f.write("\n")
            if args.dataset == "arxiv":
                f.write(
                    f"{args.backbone},"
                    + f"{args.detection_model},"
                    + f"{args.ood_year},"
                    + f"{args.K},"
                    + f"{args.T},"
                )
            elif args.dataset == "web-kb":
                f.write(
                    f"{args.backbone},"
                    + f"{args.detection_model},"
                    + f"{args.ood_data},"
                    + f"{args.K},"
                    + f"{args.T},"
                )
            elif args.dataset == "het_g":
                f.write(
                    f"{args.backbone},"
                    + f"{args.detection_model},"
                    + f"{args.ind_data},"
                    + f"{args.ood_data},"
                    + f"{args.K},"
                    + f"{args.T},"
                )
            else:
                f.write(
                    f"{args.backbone},"
                    + f"{args.detection_model},"
                    + f"{args.ood_type},"
                    + f"{args.K},"
                    + f"{args.T},"
                )
            f.write(output_info)
            f.write("\n")


if __name__ == "__main__":
    # LOAD_ARGS: 是否读取已保存初始参数
    # SAVE_INFO: 是否保存实验结果
    main(dataset="coauthor-physics", LOAD_ARGS=True, SAVE_INFO=False)
