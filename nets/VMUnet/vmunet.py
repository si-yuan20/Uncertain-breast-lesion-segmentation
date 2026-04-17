from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        
        self.input_channels = input_channels  # 保存输入通道数
        effective_in_chans = 3 if input_channels == 1 else input_channels

        self.vmunet = VSSM(in_chans=effective_in_chans,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x):
        if x.size()[1] == 1 and self.vmunet.patch_embed.proj.in_channels == 3:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")


def create_model(model_type="vmunet", num_classes=1, num_channels=1, **kwargs):
    """
    Create a VMUNet model
    
    Args:
        model_type: str, model type ("vmunet")
        num_classes: int, number of output classes
        num_channels: int, number of input channels
        **kwargs: additional parameters
    
    Returns:
        nn.Module: VMUNet model
    """
    if model_type == "vmunet":
        return VMUNet(
            input_channels=num_channels,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Test the model
if __name__ == "__main__":
    import torch
    
    # Test with 1-channel input (common in medical imaging)
    model = create_model("vmunet", num_classes=1, num_channels=1)
    input_tensor = torch.randn(2, 1, 256, 256)  # (batch_size, channels, height, width)
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("Model test completed successfully!")