import torch
import os
from monai.networks.nets import resnet18
from collections import OrderedDict

def get_medical_resnet18(checkpoint_path, num_classes=3):
    # Standard ResNet18 for 3D
    model = resnet18(spatial_dims=3, n_input_channels=1, shortcut_type='A')
    
    # Use absolute path to avoid SSH pathing issues
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Weights not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model