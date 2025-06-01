import os
import sys

import torch as tc
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, f1_score


from utils import frozen
from datautils import GroupPairActvDataset
from autoencoder import TopKSAE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
tc.cuda.set_device(int(sys.argv[1]))
SEED = int(sys.argv[2])
frozen(SEED)


class TrainingConfig:
    def __init__(self, name):
        # general settings
        self.shuffle = True
        self.seed = SEED
        self.save_path = "./outputs/%s/seed" + "%s.pt" % SEED 

        # model settings
        self.name = name
        self.in_dim = 4096
        self.l1 = 0
        self.l2 = 0
        self.dropout = 0
        
        # training settings
        self.epochs = 50
        self.bs = 32   # batch size
        self.lr = 1e-3 # learn rate
        self.grad_accumulate = 1
        self.betas = (0.9, 0.999)
        self.warmup_steps = 0
        self.decay_rate = 0.5
        self.max_decays = 2 
        self.max_tolerate = 3
        self.reload_best = True

    def to_dict(self):
        return self.__dict__



def WarmupScheduler(optimizer, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1., warmup_steps)
        return 1.0
    return tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


def DecayScheduler(optimizer, decay_rate):
    def lr_lambda(current_step):
        return decay_rate ** current_step
    return tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


def test_CLF(dataset, model):
    model.eval()
    ttls, accs = [], []
    preds, reals = [], []
    for X1, X2 in dataset:
        loss, acc, pred, real = model.compute_loss(X1.float(), X2.float())
        ttls.append(loss.item())
        accs.append(acc.item())
        preds.extend(pred)
        reals.extend(real)
    model.train()
    f1 = f1_score(reals, preds, average="macro")
    acc = accuracy_score(reals, preds)
    return acc, f1


def train_CLF(train_loader, valid_loader, test1_loader, test2_loader, model, config, loading_only=False):
    for loader in [train_loader, valid_loader, test1_loader, test2_loader]:
        loader._size = config.bs
    total_steps = len(train_loader) * config.epochs // config.grad_accumulate // config.bs
    optimizer = tc.optim.AdamW(model.parameters(), lr=config.lr, 
                              betas=config.betas, weight_decay=0.0)
    scaler = tc.amp.GradScaler("cuda", enabled=True)
    warmup_scheduler = WarmupScheduler(optimizer, config.warmup_steps,)
    decay_scheduler = DecayScheduler(optimizer, config.decay_rate,)
    bar = tqdm.tqdm(total=total_steps, desc="Training")
    best_acc, best_step = 0.0, (0, 0)
    tolerate, decays = config.max_tolerate, config.max_decays
    print("Total Size:", len(train_loader))
    for epoch in range(1, 1 + config.epochs):
        if loading_only:
            break
        ttls, accs = [], []
        for batchX, batchY in train_loader:
            batchX, batchY = batchX.float(), batchY.float()
            with tc.autocast(device_type="cuda", dtype=tc.float16, enabled=True):
                loss, acc, pred, real = model.compute_loss(batchX, batchY)
            scaler.scale(loss).backward()
            ttls.append(loss.item())
            accs.append(acc.item())
            bar.set_postfix({"Loss": np.mean(ttls),
                             "Acc": np.mean(accs)})
            
            if bar.n % config.grad_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                warmup_scheduler.step()
                bar.update(1)

        eval_loss, eval_acc = test_CLF(valid_loader, model)
        print("\nEpoch=%d | Step=%d | TrainLoss=%.4f | TrainAcc=%.4f | ValidLoss=%.4f | ValidAcc=%.4f" % (epoch, bar.n, np.mean(ttls), np.mean(accs), eval_loss, eval_acc))
        if eval_acc > best_acc + 1e-4:
            best_acc, best_step = eval_acc, (epoch, bar.n)
            model.dump_disk(config.save_path % config.name)
            tolerate = config.max_tolerate
        else:
            tolerate -= 1
            if tolerate > 0:
                continue
            if decays == 0:
                print("Early stop at Epoch=%d Step=%d" % (epoch, bar.n))
                break
            decays -= 1
            tolerate = config.max_tolerate
            decay_scheduler.step()
            model.load_disk(config.save_path % config.name)
            print("Decaying learning rate at Epoch=%d" % epoch)

    bar.close()
    if loading_only or config.reload_best:
        model.load_disk(config.save_path % config.name)
    eval_loss1, eval_acc1 = test_CLF(test1_loader, model)
    eval_loss2, eval_acc2 = test_CLF(test2_loader, model)
    print("\nBest Epoch=%d | Step=%d | ValidAcc=%.4f | Loss1=%.4f | Acc1=%.4f | Loss2=%.4f | Acc2=%.4f" % (
              best_step[0], best_step[1], best_acc, eval_loss1, eval_acc1, eval_loss2, eval_acc2))
    return model



def load_datasets(seed, layer, model="mistral"):
    train_data = GroupPairActvDataset("datasets/hh_rlhf_%s_train" % model, layer)
    valid_data = GroupPairActvDataset("datasets/hh_rlhf_%s_train" % model, layer)
    train_data.shuffle(seed=seed)
    size = int(len(train_data._data) * 0.2)
    valid_data._data = train_data._data[:size]
    train_data._data = train_data._data[size:]
    test1_data = GroupPairActvDataset("datasets/hh_rlhf_%s_valid" % model, layer)
    test2_data = GroupPairActvDataset("datasets/hh_rlhf_%s_valid" % model, layer)
    return train_data, valid_data, test1_data, test2_data


def construct_mask(fpath, l1=1e-4, l2=0.5):
    mask = [min(0, l1)] * 65536
    select = []
    with open(fpath, encoding="utf8") as f:
        f.readline()
        for row in f:
            if "Span" not in row:
                continue
            row = row.split("\t")
            if float(row[0]) >= l1 and float(row[1]) >= l2:
                mask[int(row[2])] = float(row[0])

    print("Totally %d features are selected." % sum(1 if _ > l1 else 0 for _ in mask))
    return mask

def choose_untask_mask(fpath, key="Task", task_labels={"yes", "probably"}):
    mask = [0] * 65536
    with open(fpath, encoding="utf8") as f:
        head = f.readline().strip().split("\t")
        assert head == ["FeatureID", "Task", "Verify", "Summary", "Words"]
        idx = head.index(key)
        for row in f:
            row = row.split("\t")
            if "span" not in row[-1].lower():
                continue
            if "cannot tell" in row[-2].lower():
                continue
            label = row[idx].split("|||")[0].replace("[", "").replace("]", "").strip()
            if label.strip().lower() not in task_labels: 
                mask[int(row[0])] = 1 
    print("Totally %d features are selected." % sum(mask))
    return mask

def post_check(sae, model, mask, fpath, eps=1e-4):
    Actv = sae._encode(model.linear1.weight).squeeze().cpu().tolist()
    chosen1, chosen2, chosen3 = 0, 0, 0
    with open(fpath, encoding="utf8") as f:
        f.readline()
        rows = [0] * 65536 
        for row in f:
            if row.strip() == "":
                break
            rows[int(row.split("\t")[1])] = row
        for i, (row, actv) in enumerate(zip(rows, Actv)):
            chosen3 += 1
            if actv > eps:
                chosen1 += 1
                if mask[i] == 1:
                    chosen2 += 1
    print("Total=%d | Activated=%d | Selected=%d" % (chosen3, chosen1, chosen2))

