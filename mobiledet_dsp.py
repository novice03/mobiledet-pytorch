from timm.models.efficientnet_blocks import *
from tucker_conv.conv import TuckerConv

# Target: Pixel-4 DSP
class MobileDetDSP(nn.Module):
    def __init__(self):
        super(MobileDetDSP, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU6()
        self.ibn_ne = DepthwiseSeparableConv(32, 24, act_layer = nn.ReLU6)
        
        # Second block
        self.fused1 = EdgeResidual(24, 32, exp_ratio = 4, stride = 2, act_layer = nn.ReLU6)
        self.fused2 = EdgeResidual(32, 32, exp_ratio = 4, act_layer = nn.ReLU6)
        self.ibn1 = InvertedResidual(32, 32, exp_ratio = 4)
        self.tucker1 = TuckerConv(32, 32)
        
        # Third block
        self.fused3 = EdgeResidual(32, 64, exp_ratio = 8, stride = 2, act_layer = nn.ReLU6)
        self.ibn2 = InvertedResidual(64, 64, exp_ratio = 4)
        self.fused4 = EdgeResidual(64, 64, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused5 = EdgeResidual(64, 64, exp_ratio = 4, act_layer = nn.ReLU6)
        
        # Fourth block
        self.fused6 = EdgeResidual(64, 120, exp_ratio = 8, stride = 2, act_layer = nn.ReLU6)
        self.ibn3 = InvertedResidual(120, 120, exp_ratio = 4)
        self.ibn4 = InvertedResidual(120, 120, exp_ratio = 8)
        self.ibn5 = InvertedResidual(120, 120, exp_ratio = 8)
        
        self.fused7 = EdgeResidual(120, 144, exp_ratio = 8, act_layer = nn.ReLU6)
        self.ibn6 = InvertedResidual(144, 144, exp_ratio = 8)
        self.ibn7 = InvertedResidual(144, 144, exp_ratio = 8)
        self.ibn8 = InvertedResidual(144, 144, exp_ratio = 8)
        
        # Fifth block
        self.ibn9 = InvertedResidual(144, 160, exp_ratio = 4, stride = 2)
        self.ibn10 = InvertedResidual(160, 160, exp_ratio = 4)
        self.fused8 = EdgeResidual(160, 160, exp_ratio = 4, act_layer = nn.ReLU6)
        self.tucker2 = TuckerConv(160, 160, in_comp_ratio = 0.75)        
        self.ibn11 = InvertedResidual(160, 240, exp_ratio = 8)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.ibn_ne(x)
        c1 = x
        
        # Second block
        x = self.fused1(x)
        x = self.fused2(x)
        x = self.ibn1(x)
        x = self.tucker1(x)
        c2 = x
        
        # Third block
        x = self.fused3(x)
        x = self.ibn2(x)
        x = self.fused4(x)
        x = self.fused5(x)
        c3 = x
        
        # Fourth block
        x = self.fused6(x)
        x = self.ibn3(x)
        x = self.ibn4(x)
        x = self.ibn5(x)
        x = self.fused7(x)
        x = self.ibn6(x)
        x = self.ibn7(x)
        x = self.ibn8(x)
        c4 = x
        
        # Fifth block
        x = self.ibn9(x)
        x = self.ibn10(x)
        x = self.fused8(x)
        x = self.tucker2(x)
        x = self.ibn11(x)
        c5 = x
        
        return c1, c2, c3, c4, c5       