import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__(); self.pool=nn.MaxPool2d(2)
        self.e1=DoubleConv(in_channels,base); self.e2=DoubleConv(base,base*2); self.e3=DoubleConv(base*2,base*4); self.e4=DoubleConv(base*4,base*8); self.mid=DoubleConv(base*8,base*16)
        self.up4=nn.ConvTranspose2d(base*16,base*8,2,2); self.d4=DoubleConv(base*16,base*8)
        self.up3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=DoubleConv(base*8,base*4)
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=DoubleConv(base*4,base*2)
        self.up1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=DoubleConv(base*2,base); self.out=nn.Conv2d(base,out_channels,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); d=self.e4(self.pool(c)); m=self.mid(self.pool(d))
        x=self.d4(torch.cat([self.up4(m),d],1)); x=self.d3(torch.cat([self.up3(x),c],1)); x=self.d2(torch.cat([self.up2(x),b],1)); x=self.d1(torch.cat([self.up1(x),a],1))
        return self.out(x)
