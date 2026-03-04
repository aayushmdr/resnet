import torch
import torch.nn as nn
from monai.networks.nets import resnet18

def get_medical_resnet18(checkpoint_path, num_classes=3):
    # Initialize 3D ResNet18
    model = resnet18(spatial_dims=3, n_input_channels=1, shortcut_type='A')
    
    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    state_dict = checkpoint['state_dict']
    
    # Clean keys (remove 'module.' if it exists)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    
    # Update classification head for UCEC Grades (G1, G2, G3)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model