from .models import (
    S_FC,
    FullyConnectedMNIST,
    LinearRegression,
    Net_2CNN,
    Net_2FC,
    S_Conv,
)


def get_model(model_name, config):
    if model_name == "LinearRegression":
        model = LinearRegression(
            input_dim=config.data.input_dim,
            output_dim=config.data.output_dim,
        )
    elif model_name == "FullyConnectedMNIST":
        model = FullyConnectedMNIST()
    elif model_name == "S_Conv":
        model = S_Conv(
            input_size=config.data.input_shape,
            ch=config.model.kwargs.first_out_channel,
            num_classes=10,
        )
    elif model_name == "S_FC":
        model = S_FC(
            input_size=config.data.input_shape,
            ch=config.model.kwargs.first_out_channel,
            num_classes=10,
        )
    elif model_name == "Net_2CNN":
        model = Net_2CNN(
            input_size=config.data.input_shape,
            alpha1=config.model.kwargs.first_out_channel,
            alpha2=config.model.kwargs.second_out_channel,
        )
    elif model_name == "Net_2FC":
        model = Net_2FC(
            input_size=config.data.input_shape,
            alpha1=config.model.kwargs.first_out_channel,
            alpha2=config.model.kwargs.second_out_channel,
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")

    return model
