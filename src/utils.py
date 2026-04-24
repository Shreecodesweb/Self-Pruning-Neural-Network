import torch

def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for module in model.modules():
        if hasattr(module, "get_gates"):
            gates = module.get_gates()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total


def compute_l1_loss(model):
    l1_loss = 0
    for module in model.modules():
        if hasattr(module, "get_gates"):
            l1_loss += module.get_gates().sum()
    return l1_loss


def collect_all_gates(model):
    gates_list = []
    for module in model.modules():
        if hasattr(module, "get_gates"):
            gates_list.append(module.get_gates().detach().cpu().flatten())
    return torch.cat(gates_list)
