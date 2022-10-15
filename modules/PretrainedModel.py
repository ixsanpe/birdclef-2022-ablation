import timm
from torch import nn

class PretrainedModel(nn.Module):
    def __init__(
        self, 
        model_name, 
        pretrained=True, 
        global_pool='',
        in_chans=None, 
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0, # this way, the model has no output head
            global_pool=global_pool,
            in_chans=in_chans,
        )

        if "efficientnet" in model_name:
            self.backbone_out = self.backbone.num_features
        else:
            self.backbone_out = self.backbone.feature_info[-1]["num_chs"]
 
        self.bn0 = nn.InstanceNorm2d(1) # TODO: not sure if we need this?
    
    def get_out_dim(self):
        return self.backbone_out

    def forward(self, x):
        while x.dim() < 3:
            x = x.unsqueeze(-2)
        x = x.permute(0, 2, 1)
        x = x[:, None, :, :] # add extra dimension
        x = self.bn0(x)

        x = self.backbone(x)  # (bs, channels, feats, time)
        x = x.mean(3)  # pool freq
        x = x.permute(0, 2, 1)  # bs, time, feats

        return x

class OutputHead(nn.Module):
    def __init__(
        self, 
        n_in, 
        n_out, 
        activation=nn.Sigmoid()
    ):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation 
    
    def forward(self, x, return_logits=True):
        x = self.layer(x)
        if return_logits: return x 
        else: return self.activation(x)

    

