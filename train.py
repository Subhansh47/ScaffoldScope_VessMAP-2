import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from scaffoldscope.model import UNet

ROOT=Path(__file__).resolve().parent
IMG_DIR=ROOT/'data'/'vessmap'/'images'; MASK_DIR=ROOT/'data'/'vessmap'/'masks'
EXTS={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
class VesselDataset(Dataset):
    def __init__(self, size=256):
        self.size=size; masks={p.stem:p for p in MASK_DIR.iterdir() if p.suffix.lower() in EXTS} if MASK_DIR.exists() else {}
        self.pairs=[(p,masks[p.stem]) for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS and p.stem in masks] if IMG_DIR.exists() else []
        if not self.pairs: raise RuntimeError(f'No paired image/mask files found. Put images in {IMG_DIR} and same-named masks in {MASK_DIR}.')
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        ip,mp=self.pairs[i]
        im=Image.open(ip).convert('L').resize((self.size,self.size),Image.Resampling.BILINEAR)
        ma=Image.open(mp).convert('L').resize((self.size,self.size),Image.Resampling.NEAREST)
        x=np.asarray(im,dtype=np.float32)/255.; y=(np.asarray(ma,dtype=np.float32)>127).astype(np.float32)
        return torch.from_numpy(x[None]),torch.from_numpy(y[None])

def main():
    ap=argparse.ArgumentParser(description='Train ScaffoldScope U-Net on paired vessel images/masks')
    ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--size',type=int,default=256)
    ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--device',default='auto'); args=ap.parse_args()
    torch.manual_seed(42); np.random.seed(42); ds=VesselDataset(args.size)
    nval=max(1,int(len(ds)*.2)) if len(ds)>1 else 0; ntrain=len(ds)-nval
    if ntrain<1: raise RuntimeError('Need at least 2 image/mask pairs to train with validation.')
    train_ds,val_ds=random_split(ds,[ntrain,nval],generator=torch.Generator().manual_seed(42))
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0); val_dl=DataLoader(val_ds,batch_size=args.batch_size,num_workers=0)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    model=UNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr); bce=nn.BCEWithLogitsLoss()
    out=ROOT/'models'/'scaffoldscope_unet.pt'; out.parent.mkdir(exist_ok=True)
    best=float('inf')
    for epoch in range(1,args.epochs+1):
        model.train(); tl=[]
        for x,y in train_dl:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=bce(logits,y); loss.backward(); opt.step(); tl.append(loss.item())
        model.eval(); vl=[]
        with torch.no_grad():
            for x,y in val_dl: vl.append(bce(model(x.to(device)),y.to(device)).item())
        train_loss=float(np.mean(tl)); val_loss=float(np.mean(vl)); print(f'Epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}')
        if val_loss<best:
            best=val_loss; torch.save({'model_state_dict':model.state_dict(),'image_size':args.size,'base_channels':32,'val_loss':best},out)
            print(f'  saved best checkpoint: {out}')
    print('Training finished. Review validation performance and test on held-out images before research use.')
if __name__=='__main__': main()
