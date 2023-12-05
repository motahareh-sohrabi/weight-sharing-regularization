from dgl.ops import segment_reduce
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch_scatter
import wandb
from torch_scatter import scatter


def train_epoch(
    model,
    loss_fn,
    optimizer,
    data_loader,
    config,
    device=torch.device("cpu"),
    return_ws_list=False,
    retrain_ws_list=None,
):
    """
    Trains the model for one epoch.

    Args:
        model: torch.nn.Module
        loss_fn: torch.nn.Module
        optimizer: torch.optim.Optimizer
        data_loader: torch.utils.data.DataLoader
        device: torch.device

    Returns:
        loss: float
    """

    model.train()
    loss = None
    loss_epoch = 0.0
    acc_epoch = 0.0
    cnt = 0
    metrics = defaultdict(list)
    if return_ws_list:
        ws_list = {}
    else:
        ws_list = None
    _return_ws_list = False

    for index, (input, target) in enumerate(data_loader):
        # log the ws_metric on the last epoch
        if index == len(data_loader) - 1:
            log_ws_metric = True
            if return_ws_list:
                _return_ws_list = True
        else:
            log_ws_metric = False

        if retrain_ws_list is not None:
            print("Setting the weights using retrain_ws_list")
            set_ws(model, retrain_ws_list)

        # start_time = time.time()

        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = loss_fn(output, target)
        if config.regularization.ws_alg == "subgradient":
            reg = get_R(
                model, config.regularization.ws_lr, config.regularization.prox_layer
            )

            loss = loss + reg
            # + config.regularization.lasso_lr * torch.sum(torch.abs(param.data))

            if log_ws_metric:
                wandb.log({"R": reg})

            print("R: ", reg)
            print("Loss: ", loss)

        loss_epoch += loss.item() * input.shape[0]
        acc_epoch += (output.argmax(dim=1) == target).sum().item()
        cnt += input.shape[0]

        optimizer.zero_grad()
        loss.backward()

        lr_optimizer = optimizer.param_groups[0]["lr"]

        # optimizer.step()

        if config.regularization.beta > 0.0:
            for param in model.parameters():

                param.grad += config.regularization.lmbda * \
                    torch.sign(param.data)

        optimizer.step()

        if config.regularization.beta > 0.0:
            j = 0
            for name, param in model.named_parameters():
                param.data *= (
                    torch.abs(param.data)
                    >= config.regularization.beta * config.regularization.lmbda
                )

                if log_ws_metric and name in [
                    "1.weight",
                    "1.bias",
                    "4.weight",
                    "4.bias",
                ]:
                    sparsity = (
                        torch.sum(
                            torch.abs(param.data)
                            < config.regularization.beta * config.regularization.lmbda
                        ).item()
                        / param.data.numel()
                    )
                    if len(param.data.shape) == 2:
                        metrics[f"Weight Layer_{j+1}"] = (0.0, sparsity)
                    elif len(param.data.shape) == 1:
                        metrics[f"Bias Layer_{j+1}"] = (0.0, sparsity)
                        j += 1

        corrected_beta_lasso = False

        if corrected_beta_lasso and config.regularization.beta > 0.0:
            for param in model.parameters():
                param.data *= (
                    torch.abs(param.data)
                    >= lr_optimizer
                    * config.regularization.beta
                    * config.regularization.lmbda
                )

                param.data += (
                    -lr_optimizer * config.regularization.lmbda *
                    torch.sign(param.data)
                )

        # Proximal
        if config.regularization.ws_alg == "subgradient":
            continue

        is_ws_lr_zero = (
            isinstance(config.regularization.ws_lr, float)
            and config.regularization.ws_lr == 0
        ) or (
            isinstance(config.regularization.ws_lr, tuple)
            and all(e == 0 for e in config.regularization.ws_lr)
        )
        if is_ws_lr_zero and config.regularization.lasso_lr == 0:
            continue

        if config.regularization.prox_layer == "FC1":
            try:
                first_linear_layer = get_first_layer(model)
                layers = [first_linear_layer]
            except:
                raise ValueError("No linear layer found in the model")
        elif config.regularization.prox_layer == "FC12":
            layers = [model.fc1, model.fc2]

        # if ws_lr is not a tuple then use the same ws_lr for all layers
        if not isinstance(config.regularization.ws_lr, tuple):
            ws_lr_list = [config.regularization.ws_lr] * len(layers)
        else:
            ws_lr_list = config.regularization.ws_lr

        if not config.regularization.use_trick:
            ws_lr_list = [ws_lr * lr_optimizer for ws_lr in ws_lr_list]
            lr_opt = 1.0
        else:
            lr_opt = lr_optimizer

        for i, layer in enumerate(layers):
            weights = layer.weight
            bias = layer.bias

            flatten_weight = weights.data.flatten()
            flatten_bias = bias.data.flatten()

            flatten_weight, metrics_w, ws_list_w = prox_R_and_l1(
                flatten_weight,
                step_size_R=ws_lr_list[i],
                step_size_l1=config.regularization.lasso_lr,
                rho=config.regularization.rho,
                lr=lr_opt,
                log_ws_metric=log_ws_metric,
                return_ws_list=_return_ws_list,
                alg=config.regularization.ws_alg,
            )
            flatten_bias, metrics_b, ws_list_b = prox_R_and_l1(
                flatten_bias,
                step_size_R=ws_lr_list[i],
                step_size_l1=config.regularization.lasso_lr,
                rho=config.regularization.rho,
                lr=lr_opt,
                log_ws_metric=log_ws_metric,
                return_ws_list=_return_ws_list,
                alg=config.regularization.ws_alg,
            )

            weights.data = flatten_weight.reshape(weights.data.shape)
            bias.data = flatten_bias.reshape(bias.data.shape)

            if log_ws_metric:
                metrics[f"Weight Layer_{i+1}"] = metrics_w
                metrics[f"Bias Layer_{i+1}"] = metrics_b
            if return_ws_list:
                ws_list[f"Weight Layer_{i+1}"] = ws_list_w
                ws_list[f"Bias Layer_{i+1}"] = ws_list_b

            # if log_ws_metric and config.regularization.ws_alg == "imminent_collisions":
            #     for m in metrics:
            #         if metrics[m][0] < 0.25:
            #             config.regularization.ws_alg = "search_collisions"
            #             print("Switching to search collisions")

        # print(f"One iteration took {time.time() - start_time} seconds")

    loss_epoch = loss_epoch / cnt
    acc_epoch = acc_epoch / cnt

    return loss_epoch, acc_epoch * 100.0, metrics, ws_list


def get_R(model, ws_lr, prox_layer="FC1"):

    R_val = 0.0

    is_ws_lr_zero = (isinstance(ws_lr, float) and ws_lr == 0) or (
        isinstance(ws_lr, tuple) and all(e == 0 for e in ws_lr)
    )

    if is_ws_lr_zero:
        return R_val

    if prox_layer == "FC1":
        try:
            first_linear_layer = get_first_layer(model)
            layers = [first_linear_layer]
        except:
            raise ValueError("No linear layer found in the model")
    elif prox_layer == "FC12":
        layers = [model.fc1, model.fc2]

    if not isinstance(ws_lr, tuple):
        ws_lr_list = [ws_lr] * len(layers)
    else:
        ws_lr_list = ws_lr

    for i, layer in enumerate(layers):
        weights = layer.weight
        bias = layer.bias

        flatten_weight = weights.view(-1)
        flatten_bias = bias.view(-1)

        R_val += R(flatten_weight) * ws_lr_list[i]
        R_val += R(flatten_bias) * ws_lr_list[i]

    return R_val


def R(w):
    n = w.shape[0]
    device = w.device
    x, _ = torch.sort(w)
    s = torch.cumsum(x, dim=0)
    return torch.sum(torch.arange(1, n + 1, device=device) * x - s) / (n - 1)


def get_weight_sharing(m, indices):
    m = m[m > 0]
    assert torch.sum(m) == len(indices)
    sorted_indices = torch.repeat_interleave(
        torch.arange(len(m), device=indices.device), m
    )
    cluster_indices = torch.empty_like(indices)
    cluster_indices[indices] = sorted_indices
    return cluster_indices


def do_weight_sharing(w, cluster_indices):
    with torch.no_grad():
        w_ = torch.zeros(torch.max(cluster_indices) + 1,
                         device=w.device, dtype=w.dtype)
        torch_scatter.scatter(w, cluster_indices, reduce="mean", out=w_)
        torch.gather(w_, 0, cluster_indices, out=w)
        return w


def set_ws(model, ws_lists):
    layers_ws = []
    for l in ws_lists:
        layers_ws.append(ws_lists[l])

    # print(layers_ws)

    i = 0
    for layer in [model.fc1, model.fc2]:
        w_f = layer.weight.data.flatten()
        b_f = layer.bias.data.flatten()
        layer.weight.data = do_weight_sharing(w_f, layers_ws[i]).reshape(
            layer.weight.data.shape
        )
        layer.bias.data = do_weight_sharing(b_f, layers_ws[i + 1]).reshape(
            layer.bias.data.shape
        )
        i += 2


def prox_R_and_l1(
    w,
    step_size_R,
    step_size_l1,
    rho,
    lr,
    alg="search_collisions",
    log_ws_metric=False,
    return_ws_list=False,
):
    n = w.shape[0]
    # print('Doing %s on vector of size %d...' % (alg, n))

    device = w.device

    if step_size_R > 0:
        # sort
        x, indices = torch.sort(w)

        # apply prox_R
        v = step_size_R * \
            (n - 2 * torch.arange(n, device=device) - 1) / (n - 1)
        m = torch.ones(n, dtype=torch.long, device=device)

        if alg == "search_collisions":
            x, v, m = search_collisions_alg(x, v, m)
        elif alg == "end_collisions":
            x, v, m = end_collisions_alg(x, v, m)
        elif alg == "imminent_collisions":
            x, v, m = imminent_collisions_alg_2(x, v, m)
        elif alg == "pool_adjacent_violators":
            x, v, m = torch_pool_adjacent_violators(x, v, m)

        # get weight sharing
        ws_list = get_weight_sharing(m, indices) if return_ws_list else None
    else:
        x = w
        v = torch.zeros_like(w, device=device)
        m = torch.ones(n, dtype=torch.long, device=device)
        ws_list = None

    # x += (1 - rho) * v
    # apply prox_l1
    zero_idx = torch.abs(x + v) < step_size_l1
    v -= torch.sign(x + v) * step_size_l1
    x[zero_idx] = 0
    v[zero_idx] = 0

    # update position
    x += (1 - rho) * v * lr

    # repeat particle #i m_i times
    x = torch.repeat_interleave(x, m)

    if log_ws_metric:
        num_zero_weights = torch.sum(m[zero_idx]).item()
        num_distinct_weights = torch.sum(m[~zero_idx] > 0).item() + (
            1 if num_zero_weights > 0 else 0
        )
        print(
            f"num_distinct_weights: {num_distinct_weights}, num_whole_weights: {n}, num_zero_weights: {num_zero_weights}"
        )
        num_distinct_weights /= n
        num_zero_weights /= n
        # wandb.log({f"Distinct Weights_{n}": num_distinct_weights})
    else:
        num_distinct_weights = None
        num_zero_weights = None

    if step_size_R > 0:
        # unsort
        w[indices] = x
    else:
        w = x

    return w, (num_distinct_weights, num_zero_weights), ws_list


def end_collisions_alg(x, v, m):
    n = x.shape[0]
    i = 0
    while i < n - 1:
        avg = torch.cumsum((x + v)[i:], dim=0) / torch.cumsum(m[i:], dim=0)
        j = torch.argmin(avg) + i
        if i == j:
            i += 1
            continue
        x[i] = torch.mean(x[i: j + 1])
        v[i] = torch.mean(v[i: j + 1])
        m[i] = torch.sum(m[i: j + 1])
        m[i + 1: j + 1] = 0
        i = j + 1
    return x, v, m



def imminent_collisions_step(x, v, m):
    device = x.device
    n = x.shape[0]
    if n == 1:
        return x, v, m
    c = torch.zeros(n, dtype=bool, device=device)
    c[1:] = (x + v)[:-1] < (x + v)[1:]
    i = torch.cumsum(c.to(torch.long), dim=0)
    x = scatter(x * m, i, dim=0, reduce="sum")
    v = scatter(v * m, i, dim=0, reduce="sum")
    m = scatter(m, i, dim=0, reduce="sum")
    x /= m
    v /= m
    return x, v, m


def imminent_collisions_alg(x, v, m):
    # iteratively detect and perform imminent collisions in parallel
    while True:
        old_n = x.shape[0]
        x, v, m = imminent_collisions_step(x, v, m)
        if x.shape[0] == old_n:
            break
    return x, v, m


def segment_sum(x, c):
    x = torch.cumsum(x, dim=0)
    x = x[~c]
    if x.shape[0] > 1:
        x[1:] = x[1:] - x[:-1]
    return x


def imminent_collisions_step_2(x, v, m):
    n = x.shape[0]
    if n == 1:
        return x, v, m
    c = torch.zeros(n, dtype=bool, device=x.device, requires_grad=False)
    c[:-1] = (x + v)[:-1] > (x + v)[1:]
    x = segment_sum(x * m, c)
    v = segment_sum(v * m, c)
    m = segment_sum(m, c)
    x /= m
    v /= m
    return x, v, m


def imminent_collisions_alg_2(x, v, m):
    # iteratively detect and perform imminent collisions in parallel
    num_iters = 0
    while True:
        num_iters += 1
        old_n = x.shape[0]
        x, v, m = imminent_collisions_step_2(x, v, m)
        if x.shape[0] == old_n:
            break
    print('[v2] num_iters =', num_iters)
    return x, v, m


def imminent_collisions_alg(x, v, m):
    # iteratively detect and perform imminent collisions in parallel
    while True:
        old_n = x.shape[0]
        x, v, m = imminent_collisions_step(x, v, m)
        # x, v, m = imminent_collisions_step(x, v, m, even=True)
        # x, v, m = imminent_collisions_step(x, v, m, even=False)
        if x.shape[0] == old_n:
            break
    return x, v, m


def get_first_layer(model):
    # Iterate through child modules
    for child in model.children():
        if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
            return child
        # If the child is a sequential block, recurse into it
        elif isinstance(child, nn.Sequential):
            return get_first_layer(child)
    return None


def rightmost_collision_single(x, v, m):
    avg = torch.cumsum((x + v) * m, dim=0) / torch.cumsum(m, dim=0)
    avg[avg != avg] = float("inf")
    return torch.argmin(avg, dim=0)

# NEW code


def cumsum_dim1(x):
    out = torch.cumsum(x.view(-1), dim=0).view(x.shape)
    out[1:] -= out[:-1, -1][:, None]
    return out


def rightmost_collisions(x, v, m, i):
    device = x.device
    B, n = x.shape[:2]

    idx = torch.arange(n, device=device)

    # Calculate the mask only once
    mask = idx[None, :] < i[:, None]

    # Directly calculate avg with masked operations
    masked_m = torch.where(mask, torch.zeros_like(m), m)
    cumsum_m = cumsum_dim1(masked_m)
    avg = torch.where(cumsum_m == 0, float('inf'),
                      cumsum_dim1((x + v) * masked_m) / cumsum_m)
    return torch.argmin(avg, dim=1)


def merge(x, v, m):
    device = x.device
    B = x.shape[0]
    n = x.shape[1]
    if n == 2:
        c = (x[:, 0] + v[:, 0] > x[:, 1] + v[:, 1]) & (m[:, 1] > 0)
        x[c, 0] = (x[c, 0] + x[c, 1]) / 2
        v[c, 0] = (v[c, 0] + v[c, 1]) / 2
        m[c, 0] = 2
        m[c, 1] = 0
        return
    # initialize search range
    le = torch.zeros(B, dtype=torch.long, device=device)
    ri = (n // 2) * torch.ones(B, dtype=torch.long, device=device)
    logn = n.bit_length() - 2
    for _ in range(logn):
        mid = (le + ri - 1) // 2
        j = rightmost_collisions(x, v, m, mid)
        c = (j >= n // 2)
        le[~c] = mid[~c] + 1
        ri[c] = mid[c] + 1
    i = le
    j = rightmost_collisions(x, v, m, i)
    c = (i < n // 2) & (j >= n // 2)
    if ~c.any():
        return
    m_ = m.clone()
    n = x.shape[1]
    idx = torch.arange(n, device=device)
    mask = (idx[None, :] >= i[:, None]) & (idx[None, :] <= j[:, None])
    mask[~c] = 0
    m_[~mask] = 0

    m_total = torch.sum(m_[c], dim=1)
    x[c, i[c]] = torch.sum(x[c] * m_[c], dim=1) / m_total
    v[c, i[c]] = torch.sum(v[c] * m_[c], dim=1) / m_total
    m[c, i[c]] = m_total
    mask[c, i[c]] = 0
    m[mask] = 0


def search_collisions_alg(x, v, m):
    # extend x, v, m to power of 2 length
    device = x.device
    n = x.shape[0]
    logn = (n - 1).bit_length()
    if n & (n - 1) != 0:
        n_ = 2 ** logn
        x = torch.concatenate((x, torch.zeros(n_ - n, device=device)))
        v = torch.concatenate((v, torch.zeros(n_ - n, device=device)))
        m = torch.concatenate(
            (m, torch.zeros(n_ - n, dtype=torch.long, device=device)))
    for i in range(logn):
        psize = 2 ** (i + 1)
        x = x.view(-1, psize)
        v = v.view(-1, psize)
        m = m.view(-1, psize)
        r = (n + psize - 1) // psize
        merge(x[:r], v[:r], m[:r])
    x = x.view(-1)[:n]
    v = v.view(-1)[:n]
    m = m.view(-1)[:n]
    return x, v, m


def pool_adjacent_violators(x_list, v_list, m_list):
    d = len(x_list)
    new_x_list = []
    new_v_list = []
    new_m_list = []
    for i in range(d):
        new_x_list.append(x_list[i])
        new_v_list.append(v_list[i])
        new_m_list.append(m_list[i])
        while len(new_x_list) >= 2 and new_x_list[-1] + new_v_list[-1] < new_x_list[-2] + new_v_list[-2]:
            new_x_list[-2] = (new_x_list[-2] * new_m_list[-2] + new_x_list[-1]
                              * new_m_list[-1]) / (new_m_list[-2] + new_m_list[-1])
            new_v_list[-2] = (new_v_list[-2] * new_m_list[-2] + new_v_list[-1]
                              * new_m_list[-1]) / (new_m_list[-2] + new_m_list[-1])
            new_m_list[-2] += new_m_list[-1]
            new_x_list.pop()
            new_v_list.pop()
            new_m_list.pop()
    return new_x_list, new_v_list, new_m_list


def torch_pool_adjacent_violators(x, v, m):
    device = x.device
    x_list = x.cpu().tolist()
    v_list = v.cpu().tolist()
    m_list = m.cpu().tolist()
    time0 = time.time()
    new_x_list, new_v_list, new_m_list = pool_adjacent_violators(
        x_list, v_list, m_list)
    time1 = time.time()
    print('[pava] time', time1 - time0)
    new_x = torch.tensor(new_x_list, device=device)
    new_v = torch.tensor(new_v_list, device=device)
    new_m = torch.tensor(new_m_list, device=device, dtype=torch.long)
    return new_x, new_v, new_m




def rho_scheduler(epoch, max_epochs, initial_lr=1e-8, final_lr=1.0):
    alpha = epoch / max_epochs
    return initial_lr * (1 - alpha) + final_lr * alpha
