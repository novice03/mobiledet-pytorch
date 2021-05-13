from timm.models.efficientnet_blocks import *
from tucker_conv.conv import TuckerConv

# Target: Jetson Xavier GPU
class MobileDetGPU(nn.Module):
    def __init__(self):
        super(MobileDetGPU, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU6()
        self.tucker1 = TuckerConv(32, 16, residual = False)
        
        # Second block
        self.fused1 = EdgeResidual(16, 32, exp_ratio = 8, stride = 2, act_layer = nn.ReLU6)
        self.tucker2 = TuckerConv(32, 32, out_comp_ratio = 0.25)
        self.tucker3 = TuckerConv(32, 32, out_comp_ratio = 0.25)
        self.tucker4 = TuckerConv(32, 32, out_comp_ratio = 0.25)
        
        # Third block
        self.fused2 = EdgeResidual(32, 64, exp_ratio = 8, stride = 2, act_layer = nn.ReLU6)
        self.fused3 = EdgeResidual(64, 64, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused4 = EdgeResidual(64, 64, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused5 = EdgeResidual(64, 64, exp_ratio = 4, act_layer = nn.ReLU6)
        
        # Fourth block
        self.fused6 = EdgeResidual(64, 128, exp_ratio = 8, stride = 2, act_layer = nn.ReLU6)
        self.fused7 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused8 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused9 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        
        self.fused10 = EdgeResidual(128, 128, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused11 = EdgeResidual(128, 128, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused12 = EdgeResidual(128, 128, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused13 = EdgeResidual(128, 128, exp_ratio = 8, act_layer = nn.ReLU6)
        
        # Fifth block
        self.fused14 = EdgeResidual(128, 128, exp_ratio = 4, stride = 2, act_layer = nn.ReLU6)
        self.fused15 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused16 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused17 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.ibn1 = InvertedResidual(128, 384, exp_ratio = 8)
    
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.tucker1(x)
        c1 = x
        
        # Second block
        x = self.fused1(x)
        x = self.tucker2(x)
        x = self.tucker3(x)
        x = self.tucker4(x)
        c2 = x
        
        # Third block
        x = self.fused2(x)
        x = self.fused3(x)
        x = self.fused4(x)
        x = self.fused5(x)
        c3 = x
        
        # Fourth block
        x = self.fused6(x)
        x = self.fused7(x)
        x = self.fused8(x)
        x = self.fused9(x)        
        x = self.fused10(x)
        x = self.fused11(x)
        x = self.fused12(x)
        x = self.fused13(x)
        c4 = x
        
        # Fifth block
        x = self.fused14(x)
        x = self.fused15(x)
        x = self.fused16(x)
        x = self.fused17(x)
        x = self.ibn1(x)
        c5 = x
        
        return c1, c2, c3, c4, c5