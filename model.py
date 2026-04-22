import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (
    add_remaining_self_loops,
    degree,
    remove_self_loops,
    scatter,
    to_undirected,
)
from torch_sparse import SparseTensor, matmul

from backbone import FAGCN, GAT, GATJK, GCN, GCNJK, H2GCN, MLP, MixHop
from utils import get_ada_edge_index_parall, ub_loss


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        if args.backbone == "gcn":
            self.encoder = GCN(
                in_channels=in_dim,
                hidden_channels=args.hidden_dim,
                out_channels=out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=args.use_bn,
            )
        elif args.backbone == "mlp":
            self.encoder = MLP(
                in_channels=in_dim,
                hidden_channels=args.hidden_dim,
                out_channels=out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.backbone == "gat":
            self.encoder = GAT(
                in_dim,
                args.hidden_dim,
                out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=args.use_bn,
            )
        elif args.backbone == "mixhop":
            self.encoder = MixHop(
                in_dim,
                args.hidden_dim,
                out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.backbone == "gcnjk":
            self.encoder = GCNJK(
                in_dim,
                args.hidden_dim,
                out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.backbone == "gatjk":
            self.encoder = GATJK(
                in_dim,
                args.hidden_dim,
                out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.backbone == "h2gcn":
            self.encoder = H2GCN(
                in_dim,
                args.hidden_dim,
                out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=args.use_bn,
            )
        elif args.backbone == "fagcn":
            self.encoder = FAGCN(
                in_dim,
                args.hidden_dim,
                out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        else:
            raise NotImplementedError
        self.m_criterion = ub_loss()

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args, epoch):
        """return loss for training"""
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(
            device
        )
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(
            device
        )

        # get predicted logits from gnn classifier
        logits_in = self.encoder(x_in, edge_index_in)
        logits_out = self.encoder(x_out, edge_index_out)

        train_in_idx, train_ood_idx = dataset_ind.splits["train"], dataset_ood.node_idx

        # compute supervised training loss
        if args.dataset in ("proteins", "ppi"):
            sup_loss = criterion(
                logits_in[train_in_idx],
                dataset_ind.y[train_in_idx].to(device).to(torch.float),
            )
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=-1)
            sup_loss = criterion(
                pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device)
            )

        if args.use_reg:  # if use energy regularization
            if args.dataset in (
                "proteins",
                "ppi",
            ):  # for multi-label binary classification
                logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
                logits_out = torch.stack(
                    [logits_out, torch.zeros_like(logits_out)], dim=2
                )
                energy_in = -args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(
                    dim=1
                )
                energy_out = -args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(
                    dim=1
                )
            else:  # for single-label multi-class classification
                energy_in = -args.T * torch.logsumexp(logits_in / args.T, dim=-1)
                energy_out = -args.T * torch.logsumexp(logits_out / args.T, dim=-1)

            if args.use_prop:  # use energy belief propagation
                energy_in = self.propagation(
                    energy_in, edge_index_in, args.K, args.alpha
                )[train_in_idx]
                energy_out = self.propagation(
                    energy_out, edge_index_out, args.K, args.alpha
                )[train_ood_idx]
            else:
                energy_in = energy_in[train_in_idx]
                energy_out = energy_out[train_in_idx]

            # truncate to have the same length
            if energy_in.shape[0] != energy_out.shape[0]:
                min_n = min(energy_in.shape[0], energy_out.shape[0])
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]

            # compute regularization loss
            reg_loss = torch.mean(
                F.relu(energy_in - args.m_in) ** 2
                + F.relu(args.m_out - energy_out) ** 2
            )

            loss = sup_loss + args.lamda * reg_loss
        else:
            loss = sup_loss

        if args.use_UB:
            mloss_in, _ = self.m_criterion(
                _features=logits_in[train_in_idx],
                labels=dataset_ind.y[train_in_idx].squeeze(1).to(device),
                epoch=epoch,
            )
            loss = loss + 1 * mloss_in

        return loss


class GRASP(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.encoder.eval()

    def inference(self, dataset, device, args, score="Energy"):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if score == "Energy":
            _, pred = torch.max(logits, dim=1)
            score = args.T * torch.logsumexp(logits / args.T, dim=-1)
        elif score == "MSP":
            sp = torch.softmax(logits, dim=-1)
            score, pred = sp.max(dim=-1)
        return pred, score

    def detect(self, dataset_ind, dataset_ood, device, args):
        train_idx = torch.concat(
            [dataset_ind.splits["train"], dataset_ind.splits["valid"]]
        )
        test_id = dataset_ind.splits["test"]
        test_ood = dataset_ood.node_idx

        self.T = args.T

        _, scores = self.inference(dataset_ind, device, args)
        test_nodes = torch.concat([test_id, test_ood])
        row, col = dataset_ind.edge_index

        N = dataset_ind.num_nodes
        value = torch.ones_like(row)

        adj1 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))
        adj1 = adj1.to_device(device)
        add_nodes = select_G(scores, train_idx, test_nodes, adj1)
        scores[train_idx] = torch.where(scores[train_idx] < 1, 1.0, scores[train_idx])

        edge_index = to_undirected(dataset_ind.edge_index)
        row, col = edge_index
        d = degree(col, N).float()
        d_add = torch.zeros(N, dtype=d.dtype, device=d.device)
        d_add[add_nodes] = len(add_nodes)
        d += d_add
        d_inv = 1.0 / d.unsqueeze(1)
        d_inv = torch.nan_to_num(d_inv, nan=0.0, posinf=0.0, neginf=0.0)
        d_norm = 1.0 / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        adj = adj.to_device(device)
        d_inv = d_inv.type(scores.dtype)
        d_inv = d_inv.to(device)
        e_add = torch.zeros(N, 1, dtype=scores.dtype, device=device)
        scores = scores.unsqueeze(1)
        for k in range(1, args.K + 1):
            e_add[add_nodes] = scores[add_nodes].sum() * d_inv[add_nodes]
            scores = scores * args.alpha + (matmul(adj, scores) + 1.001 * e_add) * (
                1 - args.alpha
            )
            scores[train_idx] = torch.where(
                scores[train_idx] < 1, 1.0, scores[train_idx]
            )
            if True and k < args.K:
                add_nodes = select_G2(scores, train_idx, test_nodes, adj1, k)
        scores = scores.squeeze(1)
        return (
            scores[dataset_ind.splits["test"]],
            scores[test_ood],
        )


def select_G2(scores, train_idx, test_nodes, adj, k):
    tau2 = 50
    nodes_use = test_nodes
    if tau2 == 100:
        return nodes_use.tolist()
    K = int(tau2 / 100 * len(nodes_use))

    scores = scores.squeeze(1)
    values = scores[test_nodes].cpu()

    # get Sid and Sood
    p = 5
    thresholds1 = np.percentile(values, p)
    mask = values < thresholds1
    sood = test_nodes[mask]
    thresholds2 = np.percentile(values, 100 - p)
    mask = values > thresholds2
    sid = test_nodes[mask]

    # calculate metric to select G
    N = scores.size(0)
    id_count = torch.zeros(N)
    ood_count = torch.zeros(N)
    id_count[sid] = 1
    ood_count[sood] = 1
    device = scores.device
    id_count = id_count.unsqueeze(1).to(device)
    ood_count = ood_count.unsqueeze(1).to(device)

    for _ in range(k + 1):
        id_count = matmul(adj, id_count)
        ood_count = matmul(adj, ood_count)

    id_count = id_count.squeeze(1).cpu()
    ood_count = ood_count.squeeze(1).cpu()

    metrics = id_count[nodes_use] / (ood_count[nodes_use] + 1)

    st = "top"
    # select the top big K
    if st == "top":
        return nodes_use[np.argpartition(metrics, kth=-K)[-K:]].tolist()
    # select the top small K
    elif st == "low":
        return nodes_use[np.argpartition(metrics, kth=K)[:K]].tolist()


def select_G(scores, train_idx, test_nodes, adj):
    tau2 = 50
    if tau2 == 100:
        return train_idx.tolist()
    K = int(tau2 / 100 * len(train_idx))

    values = scores[test_nodes].cpu()

    # get Sid and Sood
    p = 5
    thresholds1 = np.percentile(values, p)
    mask = values < thresholds1
    sood = test_nodes[mask]
    thresholds2 = np.percentile(values, 100 - p)
    mask = values > thresholds2
    sid = test_nodes[mask]

    # calculate metric to select G
    N = scores.size(0)
    id_count = torch.zeros(N)
    ood_count = torch.zeros(N)
    id_count[sid] = 1
    ood_count[sood] = 1
    device = scores.device
    id_count = id_count.unsqueeze(1).to(device)
    ood_count = ood_count.unsqueeze(1).to(device)
    id_count = matmul(adj, id_count)
    ood_count = matmul(adj, ood_count)

    id_count = id_count.squeeze(1).cpu()
    ood_count = ood_count.squeeze(1).cpu()

    metrics = id_count[train_idx] / (ood_count[train_idx] + 1)

    st = "top"
    # select the top big K
    if st == "top":
        return train_idx[np.argpartition(metrics, kth=-K)[-K:]].tolist()
    # select the top small K
    elif st == "low":
        return train_idx[np.argpartition(metrics, kth=K)[:K]].tolist()


class EPGNN(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.encoder.eval()

    def get_energy(self, dataset, device, args):
        """return negative energy, a vector for all input nodes"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ("proteins", "ppi"):  # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:  # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        return neg_energy, logits

    def EPGNN_propagation(self, e, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        edge_index = edge_index.cpu()
        row, col = edge_index
        e_row, e_col = e[row], e[col]
        e_ij = 1 - torch.abs(e_row - e_col)
        e_atten_fenmu = scatter(torch.exp(e_ij), row, reduce="sum")
        e_atten = torch.exp(e_ij) / e_atten_fenmu[row]
        e_j = e_atten * e_col
        e = scatter(e_j, row, reduce="sum")
        return e

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        """energy belief propagation, return the energy after propagation"""
        e = e.unsqueeze(1)
        for _ in range(prop_layers):
            e = self.EPGNN_propagation(e, edge_index)
        return e.squeeze(1)

    def detect(self, dataset_ind, dataset_ood, device, args):
        ind_e, _ = self.get_energy(dataset_ind, device, args)
        ind_e = ind_e.cpu()
        test_ood_e, _ = self.get_energy(dataset_ood, device, args)
        test_ood_e = test_ood_e.cpu()

        if args.use_prop:
            ind_e = self.propagation(ind_e, dataset_ind.edge_index, args.K, args.alpha)
            test_ood_e = self.propagation(
                test_ood_e, dataset_ood.edge_index, args.K, args.alpha
            )  # edge_index_remove_edge 表示去除ind-ood边后的edge_index
        return (
            ind_e[dataset_ind.splits["test"]],
            test_ood_e[dataset_ood.node_idx],
        )


class GraphSafe(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.encoder.eval()

    def get_energy(self, dataset, device, args):
        """return negative energy, a vector for all input nodes"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ("proteins", "ppi"):  # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:  # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        return neg_energy, logits

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5, args=None):
        """energy belief propagation, return the energy after propagation"""
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        if args.dataset == "arxiv":
            d_norm = 1.0 / torch.sqrt(d[col] * d[row])
        else:
            d_norm = 1.0 / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        adj = adj.cpu()
        # print(e.shape)
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
            # e = matmul(adj, e)
        return e.squeeze(1)

    def detect(self, dataset_ind, dataset_ood, device, args):
        ind_e, _ = self.get_energy(dataset_ind, device, args)
        ind_e = ind_e.cpu()
        test_ood_e, _ = self.get_energy(dataset_ood, device, args)
        test_ood_e = test_ood_e.cpu()

        if args.use_prop:
            ind_e = self.propagation(
                ind_e, dataset_ind.edge_index, args.K, args.alpha, args
            )
            test_ood_e = self.propagation(
                test_ood_e, dataset_ood.edge_index, args.K, args.alpha, args
            )  # edge_index_remove_edge 表示去除ind-ood边后的edge_index
        return (
            ind_e[dataset_ind.splits["test"]],
            test_ood_e[dataset_ood.node_idx],
        )


class DualChanEnergy(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.encoder.eval()

    def get_energy(self, dataset, device, args):
        """return negative energy, a vector for all input nodes"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        # logits = F.softmax(logits)
        if args.dataset in ("proteins", "ppi"):  # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:  # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        return neg_energy, logits

    def EPGNN_propagation(self, e, edge_index, edge_value):
        edge_index = edge_index.cpu()
        row, col = edge_index
        e_row, e_col = e[row], e[col]
        e_ij = torch.abs(e_row - e_col)
        e_atten_fenmu = scatter(e_ij, row, reduce="sum")
        e_atten = e_ij / e_atten_fenmu[row]
        e_atten = e_atten.squeeze()
        edge_value *= e_atten
        return edge_value

    def propagation(
        self, e, edge_index, edge_value=None, prop_layers=1, alpha=0.5, args=None
    ):
        """energy belief propagation, return the energy after propagation"""
        N = e.shape[0]

        if args.dataset == "arxiv":
            edge_index, edge_value = add_remaining_self_loops(edge_index, edge_value)
            # edge_index, edge_value = edge_index, edge_value
        else:
            edge_index, edge_value = remove_self_loops(edge_index, edge_value)
        row, col = edge_index
        d = degree(col, N).float()
        if args.dataset == "arxiv":
            d_norm = 1.0 / torch.sqrt(d[col] * d[row])
        else:
            d_norm = 1.0 / d[col]
        if edge_value is None:
            value = torch.ones_like(row)
        else:
            value = edge_value
        value *= d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        adj = adj.cpu()
        for _ in range(1):
            if args.dataset == "arxiv":
                e = matmul(adj, e)
            else:
                e = e * alpha + matmul(adj, e)
        return e.squeeze(1)

    def run_ada_energy(
        self,
        test_ind_e,
        test_ood_e,
        dataset_ind,
        dataset_ood,
        args,
        threshold=None,
        threshold_division=5,
        logits=None,
    ):
        if threshold is not None:
            threshold_ind, threshold_ood = threshold[0], threshold[1]
        else:
            # 确定阈值
            test_node_e = torch.cat(
                [test_ind_e[dataset_ind.splits["test"]], test_ood_e], dim=0
            )
            threshold_ind = np.percentile(test_node_e, 100 - threshold_division)
            threshold_ood = np.percentile(test_node_e, threshold_division)

        # for test_ind
        ind_mask_for_test_ind = test_ind_e >= threshold_ind
        ood_mask_for_test_ind = test_ind_e <= threshold_ood
        uncertain_mask_for_test_ind = (test_ind_e < threshold_ind) & (
            test_ind_e > threshold_ood
        )
        # get edge_index value for ind data
        edge_value_for_ind = get_ada_edge_index_parall(
            data=dataset_ind,
            uncertain_mask=uncertain_mask_for_test_ind,
            ind_mask=ind_mask_for_test_ind,
            ood_mask=ood_mask_for_test_ind,
            h=logits[0],
            knn_k=args.knn_k,
        )

        if args.energy_attention:
            edge_value_for_ind = self.EPGNN_propagation(
                test_ind_e, dataset_ind.edge_index, edge_value_for_ind
            )

        # for test_ood
        ind_mask_for_test_ood = test_ood_e >= threshold_ind
        ood_mask_for_test_ood = test_ood_e <= threshold_ood
        uncertain_mask_for_test_ood = (test_ood_e < threshold_ind) & (
            test_ood_e > threshold_ood
        )
        # get edge_index value for ood data
        edge_value_for_ood = get_ada_edge_index_parall(
            data=dataset_ood,
            uncertain_mask=uncertain_mask_for_test_ood,
            ind_mask=ind_mask_for_test_ood,
            ood_mask=ood_mask_for_test_ood,
            h=logits[1],
            knn_k=args.knn_k,
        )

        if args.energy_attention:
            edge_value_for_ood = self.EPGNN_propagation(
                test_ood_e, dataset_ood.edge_index, edge_value_for_ood
            )

        if args.use_prop:
            ind_e = self.propagation(
                e=test_ind_e,
                edge_index=dataset_ind.edge_index,
                edge_value=edge_value_for_ind,
                prop_layers=args.K,
                alpha=args.alpha,
                args=args,
            )
            test_ood_e = self.propagation(
                e=test_ood_e,
                edge_index=dataset_ood.edge_index,
                edge_value=edge_value_for_ood,
                prop_layers=args.K,
                alpha=args.alpha,
                args=args,
            )
            # print(test_ood_e)
        return ind_e, test_ood_e

    def detect(
        self,
        dataset_ind,
        dataset_ood,
        device,
        args,
        threshold=None,
    ):
        ind_e, logits_ind = self.get_energy(dataset_ind, device, args)
        test_ind_e = ind_e.unsqueeze(1).cpu()
        logits_ind = logits_ind.cpu()

        test_ood_e, logits_ood = self.get_energy(dataset_ood, device, args)
        test_ood_e = test_ood_e.unsqueeze(1).cpu()
        logits_ood = logits_ood.cpu()

        dataset_ind = dataset_ind.cpu()
        dataset_ood = dataset_ood.cpu()

        for _ in range(args.K):
            ind_e, test_ood_e = self.run_ada_energy(
                test_ind_e=test_ind_e,
                test_ood_e=test_ood_e,
                dataset_ind=dataset_ind,
                dataset_ood=dataset_ood,
                args=args,
                threshold=threshold,
                threshold_division=args.threshold_division,
                logits=[logits_ind, logits_ood],
            )
            test_ind_e = ind_e.unsqueeze(1).cpu()
            test_ood_e = test_ood_e.unsqueeze(1).cpu()

        test_ood_e = test_ood_e.squeeze()
        # np.savetxt("test_ind_origin_e.txt", test_ind_origin_e.numpy(), fmt="%.4f")
        # np.savetxt("test_ood_origin_e.txt", test_ood_origin_e.numpy(), fmt="%.4f")
        return (
            ind_e[dataset_ind.splits["test"]],
            test_ood_e[dataset_ood.node_idx],
        )
