# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SeResNext50, DPN68, DPN92


class MultiModel(nn.Module):
    def __init__(self, models, out_channels, pretrained=False):
        super().__init__()
        self.model_lib = {
            'seresnext50': SeResNext50,
            'dpn68': DPN68,
            'dpn92': DPN92,
        }
        
        self.img_model = self.model_lib[models[0]](3, out_channels[0], pretrained)
        
        self.vis_model = self.model_lib[models[1]](7, out_channels[1], pretrained)
        
        self.fc =  nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(out_channels[0]+out_channels[1], 128, bias=True),
                nn.Dropout(0.5),
                nn.Linear(128, 9, bias=True),
            )

    def forward(self, x_img, x_vis):
        x_img = self.img_model(x_img)
        x_vis = self.vis_model(x_vis)
        
        x = torch.cat((x_img, x_vis), dim=1)
        fea = x
        x = self.fc(x)
        
        return x, fea

class SigModel(nn.Module):
    def __init__(self, model, out_channels, pretrained=False):
        super().__init__()
        self.model_lib = {
            'seresnext50': SeResNext50,
            'dpn68': DPN68,
            'dpn92': DPN92,
        }
        
        self.model = self.model_lib[model](4, out_channels, pretrained)
        
        self.fc =  nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(out_channels, 9, bias=True),
            )

    def forward(self, x_img, x_vis):
        # N,7,26,24整型为N,1,56,78
        x_vis = x_vis.reshape(x_vis.size(0), 1, 56, -1)
        # pad为N,1,100,100
        x_vis = nn.ConstantPad2d((11,11,22,22), 0)(x_vis)
        
        x = torch.cat((x_img, x_vis), dim=1)
        
        x = self.model(x)
        
        fea = x
        x = self.fc(x)
        
        return x, fea


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    img_depth = 3
    img_height = 100
    img_width = 100
    visit_depth = 7
    visit_height = 26
    visit_width = 24
    
#    net = MultiModel(['seresnext50', 'seresnext50'], [128, 64]).to(device)
    net = SigModel('seresnext50', 128).to(device)
    
    from torchsummary import summary
    summary(net, [(img_depth, img_height, img_width), (visit_depth, visit_height, visit_width)])
    
    bs = 1
    test_x1 = torch.rand(bs, img_depth, img_height, img_width).to(device)
    test_x2 = torch.rand(bs, visit_depth, visit_height, visit_width).to(device)

    out_x, out_fea = net(test_x1, test_x2)
    print(out_x.shape)
    print(out_fea.shape)
