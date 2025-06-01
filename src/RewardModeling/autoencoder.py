import os
import numpy as np
import torch as tc



def normalized_l2(x, y):
    return (((x.to(y.device) - y) ** 2).mean(dim=1) / (y ** 2).mean(dim=1)).mean()


def layer_norm(x, eps=1e-8):
    avg = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + eps
    return (x - avg) / std, {"mu": avg, "std": std}


def MaskTopK(x, k):
    val, idx = tc.topk(x, k=k, dim=-1)
    return tc.zeros_like(x).scatter_(-1, idx ,1)


class SparseAutoencoder(tc.nn.Module):
    def __init__(self, d_inp, d_hide, device="cuda"):
        super().__init__()
        self.monitoring = False
        self.do_edit = False
        self.dims = (d_inp, d_hide)
        self.lamda = 0.1
        self.usingmean = False
        self.mask = tc.nn.Parameter(tc.ones(d_hide), requires_grad=False)
        weight = tc.nn.init.kaiming_normal_(tc.zeros(d_inp, d_hide),
                                            mode="fan_out", nonlinearity="relu")
        self.W_enc = tc.nn.Parameter(weight, requires_grad=True)
        self.b_enc = tc.nn.Parameter(tc.zeros(d_hide))
        self.b_dec = tc.nn.Parameter(tc.zeros(d_inp))
        self.freq = tc.zeros(d_hide).to(device)
        self.deads = tc.zeros(d_hide).to(device)
        self.device = device
        self.to(device)
    
    def reset_frequency(self):
        self.freq = tc.zeros_like(self.freq)

    @property
    def W_dec(self):
        return self.W_enc.T

    def _decode(self, h):
        return h @ self.W_dec + self.b_dec

    def encode(self, x):
        self.aux_loss = 0.0
        x = x.to(self.b_dec.device) - self.b_dec
        h = x @ self.W_enc + self.b_enc
        a = tc.relu(h)
        with tc.no_grad():
            self.freq += a.reshape(-1, a.shape[-1]).norm(p=0, dim=0)
        if self.alpha > 0:
            self.recons_h = self._decode(a) 
            aux_recons_h = self._decode(h * MaskTopK(-self.freq, 1024))
            self.aux_loss = normalized_l2(aux_recons_h, x - self.recons_h)
        return a

    def decode(self, h):
        return h @ self.W_dec + self.b_dec

    def forward(self, x):
        self.recons_h = None
        self.aux_loss = 0
        h = self.encode(x)
        if self.do_edit:
            h = h * self.mask.unsqueeze(0)
        x_ = self.decode(h)
        return h, x_

    def generate(self, X):
        assert len(X.shape) == 3
        self.recons_h = None
        self.aux_loss = 0
        h = self.encode(X[:, -1])
        if self.do_edit:
            h = h * self.mask.unsqueeze(0)
        X = tc.cat([X[:, :-1], self.decode(h).unsqueeze(1)], dim=1)
        return X

    def compute_loss(self, inputs, lamda=0.1):
        actvs, recons = self(inputs)
        self.actvs = actvs
        self.l2 = normalized_l2(recons, inputs)
        self.l1 = (actvs.abs().sum(dim=1) / inputs.norm(dim=1)).mean()
        self.l0 = actvs.norm(dim=-1, p=0).mean()
        self.ttl = self.l2 + self.lamda * self.l1  
        return self.ttl, self.l2, self.l1, self.l0

    @classmethod
    def from_disk(cls, fpath, device="cuda"):
        print("Loading SAE from %s." % fpath)
        states = tc.load(fpath, weights_only=True)
        model = cls(**states["config"], device="cpu")
        model.load_state_dict(states['weight'], strict=True)
        return model.to(device)

    def dump_disk(self, fpath):
        os.makedirs(os.path.split(fpath)[0], exist_ok=True)
        tc.save({"weight": self.state_dict(), 
                 "config": {"d_inp": self.dims[0], 
                            "d_hide": self.dims[1],}
                            },
                 fpath)
        print("SAE is dumped at %s." % fpath)




class TopKSAE(SparseAutoencoder):
    def __init__(self, d_inp, d_hide, topK=20, device="cuda"):
        super().__init__(d_inp, d_hide, device)
        self.topk = topK
        self.lamda = 0
        self.alpha = 0
        self.MaskTopK = True
        self.disabled = False
        self.logging = True
        self.epsilon = 1e-6
        self.recons_h = None
        self.freq = tc.zeros(d_hide).to(device)
        self.dead_mask = tc.zeros(d_hide).to(device)
        self.norm_mask = (~self.dead_mask.bool()).float()

    def set_dead_mask(self, index):
        self.dead_mask = tc.zeros(self.dims[1]).cuda()
        for i in index:
            self.dead_mask[i] = 1
        self.norm_mask = (~self.dead_mask.bool()).float().cuda()
    
    def reset_frequency(self):
        self.freq = tc.zeros_like(self.freq)

    def _encode(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def _decode(self, h):
        return h @ self.W_dec + self.b_dec

    def encode(self, x):
        x = x.to(self.b_dec.device)
        h = self._encode(x)
        mask = MaskTopK(h, self.topk)
        with tc.no_grad():
            self.freq += mask.reshape(-1, mask.shape[-1]).sum(axis=0).to(self.freq.device)
        if self.alpha > 0:
            self.recons_h = self._decode(h * mask)
            aux_recons_h = self._decode(h * MaskTopK(-self.freq, 1024))
            self.aux_loss = normalized_l2(aux_recons_h, x - self.recons_h)
        h = h * mask
        return tc.relu(h)
    
    def decode(self, h):
        if self.recons_h is None:
            self.recons_h = self._decode(h)
        return self.recons_h
    
    def compute_loss(self, inputs):
        if len(inputs.shape) == 3 and inputs.shape[0] == 1:
            inputs = inputs.squeeze(0)
        actvs, recons = self(inputs)
        self.actvs = actvs
        self.l2 = normalized_l2(recons, inputs)
        self.l1 = (actvs.abs().sum(dim=1) / inputs.norm(dim=1)).mean()
        self.l0 = actvs.norm(dim=-1, p=0).mean()
        self.ttl = self.l2 + self.alpha * self.aux_loss
        return self.ttl, self.l2, self.l1, self.l0
    
    def compute_finetune(self, x):
        x = x.to(self.b_dec.device)
        h = self._encode(x)
        actvs = tc.relu(h * MaskTopK(h.to(self.norm_mask.device) * self.norm_mask, self.topk))
        r = x - self._decode(actvs).detach()
        actvs = tc.relu(h * MaskTopK(h * self.dead_mask, self.topk))
        self.l2 = normalized_l2(self._decode(actvs), r)
        self.l1 = (actvs.abs().sum(dim=1) / x.norm(dim=1)).mean()
        self.l0 = actvs.norm(dim=-1, p=0).mean()
        self.ttl = self.l2
        return self.ttl, self.l2, self.l1, self.l0

    
    def dump_disk(self, fpath):
        os.makedirs(os.path.split(fpath)[0], exist_ok=True)
        tc.save({"weight": self.state_dict(), 
                 "config": {"d_inp": self.dims[0], 
                            "d_hide": self.dims[1], 
                            "topK": self.topk}
                            },
                 fpath)
        print("SAE is dumped at %s." % fpath)



SAEs = {"sae": SparseAutoencoder, "ae": SparseAutoencoder, "topk": TopKSAE, 
        "topk5": TopKSAE, "topk6": TopKSAE, "topk7": TopKSAE}
def load_pretrained(fpath, device="cuda"):
    assert os.path.exists(fpath)
    name = os.path.split(fpath)[-1].rsplit(".pth")[0]
    cls = name.split("_l", 1)[0].lower()
    layer = int(name.split("_l", 1)[1].split("_", 1)[0])
    return name, layer, SAEs[cls].from_disk(fpath, device)
