import argparse, os, random, json, math, logging, itertools, time
from pathlib import Path
import wfdb, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler

LEADS, FS, DUR = 12, 500, 10
LEVELS, DIM_CODE = 6, 4
SEQ_LEN = FS*DUR
U_OUT_C = 32
ECG_DIM = LEADS*U_OUT_C

class Dataset_ECG_Single(Dataset):
    def __init__(self, root_path_ecg, train_path,flag,shuffle_flag,scale,num_data=None):

        self.root_path_ecg = root_path_ecg
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        if shuffle_flag:
            random.seed(42)

        if num_data:
            self.json_data = self.__load_json__(train_path,shuffle_flag)[:num_data]
        else:
            self.json_data = self.__load_json__(train_path,shuffle_flag)

        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()


    def __load_json__(self,train_path, shuffle_flag):
        with open(train_path, "r") as f:
            records = json.load(f)
        if shuffle_flag:
            random.shuffle(records)
        return records

    def __load_ecg_data__(self, root_path_ecg, filepath):
        data = wfdb.rdsamp(os.path.join(root_path_ecg, filepath))
        signal, meta = data
        return np.array(signal)

    def __getitem__(self, index):
        example = self.json_data[index]
        example_str = json.dumps(example)
        ecg_data = self.__load_ecg_data__(self.root_path_ecg, example["ecg_path"])

        if self.scale:
            self.scaler.fit(ecg_data)
            ecg_data = self.scaler.transform(ecg_data)
        else:
            ecg_data = ecg_data

        question = example["question"]
        answer_text = ".".join(example["answer"])

        return ecg_data,question,answer_text,example_str

    def __len__(self):
        return  len(self.json_data)

def gn(c):
    return nn.GroupNorm(8, c)

class InceptionBlock1Dv4(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        b = out_c//4
        self.b1 = nn.Conv1d(in_c,b,3,padding="same",bias=False)
        self.b2 = nn.Conv1d(in_c,b,5,padding="same",bias=False)
        self.b3 = nn.Conv1d(in_c,b,7,padding="same",bias=False)
        self.pool = nn.MaxPool1d(3,1,1)
        self.b4 = nn.Conv1d(in_c,b,1,bias=False)
    def forward(self,x):
        return torch.cat([self.b1(x),self.b2(x),self.b3(x),self.b4(self.pool(x))],1)

class ResInception(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.inc=InceptionBlock1Dv4(in_c,out_c)
        self.match = nn.Identity() if in_c==out_c else nn.Conv1d(in_c,out_c,1,bias=False)
        self.norm = gn(out_c)
        self.act = nn.LeakyReLU(inplace=True)
    def forward(self,x):
        y=self.inc(x)
        return self.act(self.norm(y+self.match(x)))

class ECGEncoder(nn.Module):
    def __init__(self, depth=4):
        super().__init__()
        self.blocks=nn.ModuleList()
        in_c=1
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                ResInception(in_c,U_OUT_C),
                nn.Conv1d(U_OUT_C,U_OUT_C,3,2,1,bias=False)
            ]))
            in_c=U_OUT_C
        self.bottleneck=ResInception(U_OUT_C,U_OUT_C)
        self.to_code = nn.Linear(ECG_DIM, DIM_CODE)
    def _enc(self,x):
        for res,down in self.blocks:
            x=res(x)
            x=down(x)
        return self.bottleneck(x)
    def forward(self,sig):
        sig=sig.unsqueeze(2)
        feats=[self._enc(sig[:,i]) for i in range(LEADS)]
        feat=torch.cat(feats,1)
        feat=feat.permute(0, 2, 1)
        code = self.to_code(feat)
        return code

class ECGDecoder(nn.Module):
    def __init__(self,in_dim=ECG_DIM,depth=4,dim_code=DIM_CODE):
        super().__init__()
        self.up,self.dec=nn.ModuleList(),nn.ModuleList()
        self.expand = nn.Linear(dim_code, in_dim)
        for _ in range(depth):
            self.up.append(nn.ConvTranspose1d(in_dim,in_dim,3,2,1,output_padding=1))
            self.dec.append(ResInception(in_dim,in_dim))
        self.final=nn.Conv1d(in_dim,LEADS,1)
    def forward(self,z):
        zq = self.expand(z)
        x=zq.transpose(1,2)
        for up,res in zip(self.up,self.dec): x=res(up(x))
        return self.final(x)[..., :SEQ_LEN]

class FSQ(nn.Module):
    def __init__(self, levels=LEVELS):
        super().__init__()
        self.n=levels-1
    def forward(self,x):
        x_cont=torch.sigmoid(x)
        x_disc=torch.round(x_cont*self.n)/self.n
        return x_cont, x_disc

class FSQ_AE(nn.Module):
    def __init__(self,levels=LEVELS):
        super().__init__()
        self.levels = levels
        self.enc=ECGEncoder()
        self.quant=FSQ(levels)
        self.dec=ECGDecoder()
        self.commit_w=0.25
    def forward(self,sig):
        z=self.enc(sig) # (B,T,D)
        z_cont,z_disc=self.quant(z)
        z_st=z_cont + (z_disc-z_cont).detach()
        rec=self.dec(z_st)
        recon=F.mse_loss(rec, sig)
        return rec, recon, z_disc

def validate(model,loader,device,n_levels):
    model.eval()
    tot_loss=tot_recon=0.
    n=0
    token_used=torch.zeros(n_levels,dtype=torch.bool,device=device)
    with torch.no_grad():
        for step, data in enumerate(loader, 1):
            sig, _, _, _ = data
            sig = sig.to(device).permute(0, 2, 1).to(torch.float32)
            rec, recon, z_disc=model(sig)
            loss = recon
            tot_loss+=loss.item()*sig.size(0)
            tot_recon+=recon.item()*sig.size(0)
            n+=sig.size(0)
            idx=(z_disc.flatten()* (n_levels-1)+0.5).long()
            token_used |= torch.bincount(idx,minlength=n_levels).bool()
    return tot_loss/n, tot_recon/n, token_used.float().mean().item()

@torch.no_grad()
def token_usage(disc, n_levels):
    idx = (disc.flatten() * (n_levels - 1) + 0.5).long()   # 0‥5
    used = torch.bincount(idx, minlength=n_levels).bool()
    return used

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root_ecg", default="/data/ECG-data/MIMIC-ECG/files", help="Root directory of raw ECG data files (base path for wfdb reads)")
    ap.add_argument("--train_path", default="../dataset/ecgqa/mimic-iv-ecg/template_train_background.json", help="Path to training JSON; each entry has ecg_path, question, answer, etc.")
    ap.add_argument("--batch", type=int, default=200, help="Training batch size")
    ap.add_argument("--workers", type=int, default=20, help="DataLoader num_workers (number of data-loading processes)")
    ap.add_argument("--log_root", default="./runs", help="Root directory for TensorBoard and other logs")
    ap.add_argument("--val_path", default="../dataset/ecgqa/mimic-iv-ecg/template_valid_background.json", help="Path to validation JSON")
    ap.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=2e-3, help="Optimizer learning rate")
    ap.add_argument("--val_every", type=int, default=500, help="Run validation every N training steps")
    ap.add_argument("--patience", type=int, default=3, help="Early stopping: stop after this many validations without improvement")
    args=ap.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    ds  = Dataset_ECG_Single(args.root_ecg, args.train_path, "train", shuffle_flag=True, scale=True)
    train_dl  = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_ds = Dataset_ECG_Single(args.root_ecg, args.val_path, "valid", shuffle_flag=False, scale=True, num_data=10000)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model=FSQ_AE().to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
    logdir=Path('result_tokenzier')
    logdir.mkdir(exist_ok=True)
    writer=SummaryWriter(logdir.as_posix())

    best=math.inf
    bad=0
    start=time.time()
    used_train = torch.zeros(model.levels, dtype=torch.bool, device=device)

    for ep in range(1,args.epochs+1):
        model.train()
        for step, data in enumerate(train_dl, 1):
            sig, _, _, _ = data
            sig = sig.to(device).permute(0, 2, 1).to(torch.float32)
            rec, recon, disc = model(sig)
            loss = recon
            optim.zero_grad()
            loss.backward()
            optim.step()

            used_train |= token_usage(disc, model.levels).bool()

            # logging train
            if step%50==0:
                writer.add_scalar('train/total',loss.item(),step)
                writer.add_scalar('train/recon',recon.item(),step)
                logging.info('train total loss %.4f, recon loss %.4f, step %d', loss.item(), recon.item(), step)

            # validation every N step
            if step%args.val_every==0 and step>0:
                train_util = used_train.float().mean().item()
                used_train.zero_()
                writer.add_scalar('train/token_usage', train_util, step)

                val_loss,val_recon,val_util=validate(model,val_dl,device,LEVELS)
                writer.add_scalar('val/total',val_loss,step)
                writer.add_scalar('val/recon',val_recon,step)
                writer.add_scalar('val/token_usage',val_util,step)
                logging.info(f"step {step:>7} | train {loss.item():.4f}/{recon.item():.4f}/{train_util*100:.1f}%  ―  val {val_loss:.4f}/{val_recon:.4f}/{val_util*100:.1f}%")

                # early‑stop check
                if val_loss+1e-6 < best:
                    best=val_loss
                    bad=0
                    torch.save({'model':model.state_dict(),'step':step}, logdir/'best.pt')
                else:
                    bad+=1
                    if bad>=args.patience:
                        logging.info('Early‑stop triggered.')
                        writer.close()
                        return
            step+=1
        logging.info(f"Epoch {ep} done | elapsed {time.time()-start:.1f}s")
    writer.close()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    main()