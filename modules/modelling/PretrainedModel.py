import timm
from torch import nn
import torch 

class PretrainedModel(nn.Module):
    """
    Load a pretrained model from timm by specifying a model name, whether to use the
    pretrained or randomly initialized version, what sort of pooling to use and the number
    of in_channels

    Adapted form from Henkel et. al. 
    https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/26438069466242e9154aacb9818926dba7ddc7f0
    """
    def __init__(
        self, 
        model_name: str, 
        pretrained: bool=True, 
        global_pool: str='',
        in_chans=None, 
        **kwargs,
    ):
        """
        Parameters:
            model_name:
                the model name of the desired timm model
            pretrained:
                Whether to load the pretrained model or randomly initialize parameters
            global_pool:
                parameter in timm.create_model
            in_chans:
                None or int. If specified it is passed to the initialized model 'model_name'
        """
        super().__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0, # this way, the model has no output head
            global_pool=global_pool,
            in_chans=in_chans,
            **kwargs,
        )

        if "efficientnet" in model_name:
            self.backbone_out = self.backbone.num_features
        else:
            self.backbone_out = self.backbone.feature_info[-1]["num_chs"]
 
    
    def get_out_dim(self):
        return self.backbone_out

    def forward(self, x: torch.Tensor):
        x = x.swapaxes(-1, -2)
        x = x.unsqueeze(1) # add appropriate channel; (bs, channels=1, )

        x = self.backbone(x)  
        x = x.mean(-1).mean(-1)  # pool freq
        # while x.shape[-1] == 1:
        #     x = x.squeeze(-1)
        # x = x.swapaxes(-1, -2)  # bs, time, feats

        return x

class OutputHead(nn.Module):
    """
    It is useful to split this part from the model itself in case we want to do some 
    post-processing on the model outputs before making predictions
    """
    def __init__(
        self, 
        n_in: int, 
        n_out: int, 
        activation: callable=nn.Sigmoid()
    ):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation 
    
    def forward(self, x, return_logits=True):
        x = self.layer(x)
        if return_logits: return x 
        else: return self.activation(x)

# class PretrainedTorch(nn.Module):    
#     def __init__(self) -> None:
#         super().__init__()
#         from torchvision.models import resnet50, ResNet50_Weights

#         # Old weights with accuracy 76.130%
#         model = resnet50(weights=None)
#         self.model = model 
    
#     def forward(self, x):
#         print(x.shape)
#         return self.model(x)

