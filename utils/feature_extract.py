import torch.nn as nn

def get_resent_features(models): 
    need_layers = ['conv1', 'bn1', 'relu', 'maxpool', 
                   'layer1', 'layer2', 'layer3', 'layer4']
                #    'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    features = [] 
    for name, module in models.named_modules():
        if name in need_layers:
            features.append(module)
    return nn.Sequential(*features)