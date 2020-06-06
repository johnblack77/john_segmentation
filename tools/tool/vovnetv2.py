import torch.nn as nn

class _OSA_module(nn.Module):
    def __init__(
        self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch,
                  "{}_reduction".format(module_name), "0")))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(
                    nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(
                    nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i)))
                )
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat"))
        )

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt