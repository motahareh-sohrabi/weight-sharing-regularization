import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def set_seed(seed: int):
    """Sets the seed for the random number generators used by random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def compute_conv_output_size(input_size, kernel_size, stride, padding, dilation=1):
    """Compute the output size after a convolution operation."""
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def compute_sparsity(model, layer="All"):
    total_parameters = 0
    zero_parameters = 0

    if layer == "All":
        layers = model.modules()
    elif layer == "FC1":
        layers = [get_first_layer(model)]
    elif layer == "FC12":
        layers = [model.fc1, model.fc2]
    else:
        raise ValueError("Unsupported layer type")

    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            H_in, W_in = model.input_height, model.input_width

            # Calculate output shape of the current convolution layer
            H_out = compute_conv_output_size(
                H_in, layer.kernel_size[0], layer.stride[0], layer.padding[0]
            )
            W_out = compute_conv_output_size(
                W_in, layer.kernel_size[1], layer.stride[1], layer.padding[1]
            )

            C_in, C_out = layer.in_channels, layer.out_channels
            H_f, W_f = layer.kernel_size

            zero_params_interpreted_fc = (
                H_in * W_in * C_in * H_out * W_out * C_out - C_in * C_out * H_f * W_f
            )
            total_params_interpreted_fc = H_in * W_in * C_in * H_out * W_out * C_out

            zero_parameters += zero_params_interpreted_fc
            total_parameters += total_params_interpreted_fc

            # Update H_in and W_in for the next layer
            H_in, W_in = H_out, W_out

        elif isinstance(layer, nn.Linear):
            # Traditional sparsity calculation for fully connected layers
            tolerance = torch.tensor(1e-8)
            zero_parameters += torch.sum(
                torch.isclose(layer.weight.cpu(), torch.tensor(0.0), atol=tolerance)
            )
            total_parameters += layer.weight.numel()

            if layer.bias is not None:
                total_parameters += layer.bias.numel()

    sparsity = zero_parameters / total_parameters
    return sparsity


def get_first_layer(model):
    # Iterate through child modules
    for child in model.children():
        if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
            return child
        # If the child is a sequential block, recurse into it
        elif isinstance(child, nn.Sequential):
            return get_first_layer(child)
    return None


def get_heatmap(model, map_shape, row=0):
    first_layer = get_first_layer(model)

    if first_layer is None:
        raise ValueError("Couldn't find a Linear or Conv2d layer.")

    # Check if first layer is Fully Connected (Linear)
    if isinstance(first_layer, nn.Linear):
        weight_data = first_layer.weight.data[row]
        heatmap = weight_data.reshape(map_shape)
        # Normalize for visualization
        heatmap = (heatmap - torch.min(heatmap)) / (
            torch.max(heatmap) - torch.min(heatmap)
        )
        # Convert to PIL image for visualization
        heatmap_img = transforms.ToPILImage()(heatmap)
    # Check if first layer is Convolutional
    elif isinstance(first_layer, nn.Conv2d):
        weights = first_layer.weight.data

        # Extracting the first filter's weights.
        filter_weights = weights[0]

        num_channels = weights.shape[1]
        # Normalize for visualization across each channel separately
        for i in range(num_channels):
            filter_weights[i] = (filter_weights[i] - torch.min(filter_weights[i])) / (
                torch.max(filter_weights[i]) - torch.min(filter_weights[i])
            )

        # Create an empty output image of the size of the input
        output = torch.zeros(map_shape)

        # Place the heatmap of the filter on the top-left corner of the output
        output[:, : filter_weights.shape[1], : filter_weights.shape[2]] = filter_weights

        # Convert to PIL image for visualization
        heatmap_img = transforms.ToPILImage()(output)

        # #Or just select the first channel (e.g., R channel) for visualization.
        # heatmap = filter_weights[0]

        # output = torch.zeros((model.input_height, model.input_width))

        # output[: heatmap.shape[0], : heatmap.shape[1]] = heatmap
        # output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))

        # heatmap_img = transforms.ToPILImage()(output)
    else:
        raise ValueError("Unsupported first layer type")

    return heatmap_img


def get_heatmap_fig(model, map_shape):
    first_layer = get_first_layer(model)
    if isinstance(first_layer, nn.Linear):
        w = first_layer.weight.data
        first_layer_size = w.shape[0]
        NH = 10
        NW = 10
        rng = np.random.RandomState(777)
        filter_indices = rng.choice(first_layer_size, size=(NH, NW), replace=False)

        fig = plt.figure(figsize=(NW, NH))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(NH):
            for j in range(NW):
                row = filter_indices[i, j]
                heatmap = get_heatmap(model=model, map_shape=map_shape, row=row)
                ax = fig.add_subplot(NH, NW, i * NW + j + 1)
                ax.imshow(heatmap, cmap="gray")
                ax.axis("off")

    elif isinstance(first_layer, nn.Conv2d):
        heatmap = get_heatmap(model=model, map_shape=map_shape)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(heatmap, cmap="gray")
        ax.axis("off")

    return fig


def compute_distinct_w(w):
    non_zeros = w[np.abs(w) > 1e-8]
    total_non_zeros = len(non_zeros)

    # Now, we'll use np.unique with a specified tolerance to find the unique elements.
    # This is much more efficient than manually comparing each pair of elements.

    # First, we need to round the numbers to a fixed precision that corresponds to the tolerance.
    precision = int(np.ceil(-np.log10(1e-8)))  # This corresponds to the tolerance
    rounded_weights = np.round(non_zeros, decimals=precision)

    # Use np.unique to identify distinct weights
    unique_weights = np.unique(rounded_weights)
    total_unique_weights = len(unique_weights)

    # Compute the ratio
    unique_weight_ratio = (
        total_unique_weights / total_non_zeros if total_non_zeros > 0 else 0
    )

    return unique_weight_ratio


def compute_distinct_non_zero_weights(model, layer_name="FC1"):
    # Retrieve the appropriate layer
    if layer_name == "All":
        layers = model.modules()
    elif layer_name == "FC1":
        layers = [get_first_layer(model)]
    elif layer_name == "FC12":
        layers = [model.fc1, model.fc2]
    else:
        raise ValueError("Invalid layer name. Please use 'All' or 'FC1'.")

    all_weights = []

    for layer in layers:
        if isinstance(layer, torch.nn.Linear):  # Check if the layer is fully connected
            weights = layer.weight.detach().cpu().numpy()  # Make sure it's on CPU
            all_weights.append(weights)

    # Combine weights from all relevant layers
    all_weights = np.concatenate([w.flatten() for w in all_weights])

    # Compute the distinct weight ratio
    unique_weight_ratio = compute_distinct_w(all_weights)

    return unique_weight_ratio
