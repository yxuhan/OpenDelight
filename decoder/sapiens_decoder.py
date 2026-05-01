import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def count_parameters(model: nn.Module, detailed: bool = True) -> int:
    """Count parameters in a PyTorch model.

    Args:
        model: PyTorch model to count parameters for
        detailed: whether to print per-layer details

    Returns:
        Total number of parameters in the model
    """
    from prettytable import PrettyTable
    total_params = 0
    
    if detailed:
        table = PrettyTable(["Layer Name", "Param Count", "Trainable"])
    
    for name, parameter in model.named_parameters():
        if "img_encoder" in name:
            continue
        if "vol_decoder" in name:
            continue
        param_count = parameter.numel()
        total_params += param_count
        
        if detailed:
            table.add_row([name, param_count, parameter.requires_grad])
    
    if detailed:
        print(table)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return total_params


class SapiensDecoder(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            deconv_out_channels,
            deconv_kernel_sizes,
            conv_out_channels,
            conv_kernel_sizes,
            interpolate_mode='bilinear', 
            grad_checkpointing=False,
            **kwargs):
        super().__init__()

        self.interpolate_mode = interpolate_mode
        self.grad_checkpointing = grad_checkpointing

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        self.cls_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _make_conv_layers(self, in_channels,
                          layer_out_channels,
                          layer_kernel_sizes) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(
                nn.Conv2d(**cfg)
            )
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels,
                            layer_kernel_sizes) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(
                nn.ConvTranspose2d(**cfg)
            )
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, inputs):
        if self.grad_checkpointing and self.training:
            x = checkpoint(self.deconv_layers, inputs, use_reentrant=False) ## B x 768 x 512 x 384
            x = checkpoint(self.conv_layers, x, use_reentrant=False) ## B x 768 x 512 x 384
        else:
            # inputs = self._transform_inputs(inputs) ## B x 1536 x 64 x 48
            x = self.deconv_layers(inputs) ## B x 768 x 512 x 384
            x = self.conv_layers(x) ## B x 768 x 512 x 384
            
        out = self.cls_conv(x) ## B x 1 x 512 x 384
        return out


if __name__ == "__main__":
    device = "cuda"
    
    network = SapiensDecoder(
        in_channels=512,
        out_channels=3,
        deconv_out_channels=(512, 512, 512, 512), ## this will 2x at each step. so total is 8x
        deconv_kernel_sizes=(4, 4, 4, 4),
        conv_out_channels=(512, 512, 512, 512),
        conv_kernel_sizes=(1, 1, 1, 1),
    ).to(device)
    
    feat = torch.randn(2, 512, 32, 32).to(device)
    
    with torch.no_grad():
        output = network(feat)
    print(output.shape) ## B x 1 x 256 x 192
    
    count_parameters(network, detailed=True)
