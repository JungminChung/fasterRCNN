import torch
import torch.nn as nn 
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class AAA(nn.Module):
    def __init__(self, backbone_type, backbone_pretrained, args):
        super(AAA, self).__init__()
        self.backbone_type = backbone_type
        self.backbone_pretrained = backbone_pretrained
        self._get_backbone()

        self.intra_red = IntraAttention('red', args.intra_identity, args)
        self.intra_green = IntraAttention('green', args.intra_identity, args)
        self.intra_blue = IntraAttention('blue', args.intra_identity, args)

        self.inter = InterAttention(args.inter_identity, args)

    
    def _get_backbone(self):
        if self.backbone_type == 'res18' : 
            backbone = resnet18(pretrained=self.backbone_pretrained)
        elif self.backbone_type == 'res34' : 
            backbone = resnet34(pretrained=self.backbone_pretrained)
        elif self.backbone_type == 'res50' : 
            backbone = resnet50(pretrained=self.backbone_pretrained)
        elif self.backbone_type == 'res101' : 
            backbone = resnet101(pretrained=self.backbone_pretrained)
        elif self.backbone_type == 'res152' : 
            backbone = resnet152(pretrained=self.backbone_pretrained)
        else : 
            raise RuntimeError('Invalide backbone type'
                               'it can be res18 | res34 | res50 | res101 | res152')
        
        children = nn.Sequential(*list(backbone.children()))
        
        self.bb_front = children[:4]

        self.bb_layer1 = children[4]
        self.bb_layer2 = children[5]
        self.bb_layer3 = children[6]
        self.bb_layer4 = children[7]

    def mix_inter_intra(self, intra, inter, method):
        assert method in ['avg'], 'mix_inter_intra is avg or else'
        if args.mix_inter_intra == 'avg':
            mask = (intra_x + inter_x) / 2 
        
        return mask 

    def forward(self, x): 
        # intra attention 
        red = x[:, 0, :, :].unsqueeze(1)
        green = x[:, 1, :, :].unsqueeze(1)
        blue = x[:, 2, :, :].unsqueeze(1)

        intra_x = torch.cat([self.intra_red(red), 
                             self.intra_green(green), 
                             self.intra_blue(blue)], 
                            dim=1) 
        # TODO : save images 

        # inter attention 
        inter_x = self.inter(x)
        # TODO : save images 
        
        mask = self.mix_inter_intra(intra_x, inter_x, args.mix_inter_intra)
        enhanced_x = x * mask 

        x = self.bb_front(enhanced_x)

        x = self.bb_layer1(x)
        x = self.bb_layer2(x)
        x = self.bb_layer3(x)
        x = self.bb_layer4(x)

        return x 


class IntraAttention(nn.Module):
    def __init__(self, color_type, identity, args):
        super(IntraAttention, self).__init__()
        self.color_type = color_type 
        self.identity = identity
        self.args = args

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        assert x.shape[1] == 1, "IntraAttention Module get 1 channel image"
        
        if self.identity : 
            return x 

        x = self.convs(x)
        x = self.sig(x) 
        # TODO : save image 
        return x 

class InterAttention(nn.Module):
    def __init__(self, identity, args):
        super(InterAttention, self).__init__()
        self.identity = identity 
        self.args = args 

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=3),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        assert x.shape[1] == 3, "InterAttention Module get 3 channel image"
        
        if self.identity : 
            return x 
        
        x = self.convs(x)
        x = self.sig(x) 
        # TODO : save image 
        return x 