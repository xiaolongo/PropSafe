import argparse
import copy
import os
import os.path as osp

import torch
from torch_geometric import seed_everything

from args import load_args, parser_add_main_args
from logger import Logger_classify
from model import GraphEncoder
from utils import eval_acc, evaluate_classify, load_dataset


def main(dataset: str, LOAD_ARGS: bool = True):

    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    if LOAD_ARGS:
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

    dataset_ind, dataset_ood_tr, _ = load_dataset(args)

    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if len(dataset_ood_tr.y.shape) == 1:
        dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)

    in_dim = dataset_ind.x.shape[1]
    out_dim = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])

    model = GraphEncoder(in_dim, out_dim, args).to(device)
    criterion = torch.nn.NLLLoss()
    eval_func = eval_acc

    logger = Logger_classify(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        best_val = float("-inf")
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = model.loss_compute(
                dataset_ind, dataset_ood_tr, criterion, device, args, epoch
            )
            loss.backward()
            optimizer.step()

            result = evaluate_classify(
                model, dataset_ind, eval_func, criterion, args, device
            )
            logger.add_result(run, result)

            if result[1] >= best_val:
                best_val = result[1]
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"Epoch: {epoch:02d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, "
                f"Valid: {100 * result[1]:.2f}%, "
                f"Test: {100 * result[2]:.2f}%"
            )

        logger.print_statistics(run)

        if args.dataset not in ["arxiv", "web-kb"]:
            model_dir = osp.join(
                "checkpoints",
                f"{args.dataset.lower()}",
                f"{args.backbone}_{args.ood_type}",
            )
        else:
            model_dir = osp.join(
                "checkpoints",
                f"{args.dataset.lower()}",
                f"{args.backbone}",
            )
        os.makedirs(model_dir, exist_ok=True)
        torch.save(best_state, osp.join(f"{model_dir}", f"model{run}.pt"))
        # chamelon
        # torch.save(model.state_dict(), osp.join(f"{model_dir}", f"model{run}.pt"))


if __name__ == "__main__":
    main(dataset="arxiv", LOAD_ARGS=True)
