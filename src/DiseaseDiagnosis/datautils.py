import os
import random
import time
import pickle as pkl
from queue import Queue
from threading import Thread

import tqdm
import torch as tc
import numpy as np


class ThreadPool:
    def __init__(self, call_fn, num_workers=1, max_preload=4):
        assert max_preload >= num_workers
        self._closed = True
        self._callfn = call_fn
        self._inputs = Queue()
        self._outputs = Queue(max_preload)
        self._pool = []
        self._callbacks = {}
        self._size = 0
        self._idx = 0
        self._nworkers = num_workers

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        assert isinstance(idx, int) and 0 <= idx < self._size
        while idx not in self._callbacks:
            self._callbacks.update([self._outputs.get()])
        return self._callbacks.pop(idx)

    def __iter__(self):
        while True:
            try:
                yield self[self._idx]
            except AssertionError:
                break
            self._idx += 1
        self._size = 0
        self._idx = 0

    def _call(self):
        while True:
            idx, args = self._inputs.get()
            if idx is None:
                break
            try:
                rslt = self._callfn(args)
            except Exception as e:
                rslt = e
            self._outputs.put((idx, rslt))

    def collect(self):
        return list(self)

    def submit(self, items):
        assert self._closed is False
        for idx, item in enumerate(items, self._size):
            self._inputs.put((idx, item))
            self._size += 1

    def launch(self):
        if len(self._pool) == 0 and self._closed is True:
            for _ in range(self._nworkers):
                worker = Thread(target=self._call, daemon=True)
                worker.start()
                self._pool.append(worker)
            self._closed = False

    def close(self):
        if len(self._pool) > 0 and self._closed is False:
            for _ in range(self._nworkers):
                self._inputs.put((None, None))
            self._closed = True

    def join(self):
        assert self._closed is True
        while len(self._pool) > 0:
            self._pool.pop(0).join()


class GroupPairActvDataset:
    def __init__(self, fpath, layerID, batch_size=1, workers=1, aug=0):
        fpath = fpath + '_l%d_last' % layerID
        self._root = fpath
        self._size = batch_size
        self._layer = layerID
        self._data = sorted([_ for _ in os.listdir(fpath) if _.startswith("group_") and _.endswith(".pkl")])
        self._loading_file = lambda x: pkl.load(open(fpath + '/' + x, "rb"))
        self._blocksize = len(self._loading_file(self._data[0]))
        self._pool = ThreadPool(self._loading_file, num_workers=workers, max_preload=workers*2)
        self._augsize = aug
        self._cached_files = []

    def __getitem__(self, idx):
        shift = idx // self._blocksize
        residual = idx % self._blocksize
        if shift != self.precache[0]:
            del self.precache
            begin = time.time()
            self.precache = (shift, self._loading_file(self._data[shift]))
            spend = time.time() - begin
        return self.precache[1][residual]

    def __len__(self):
        return int((1 + self._augsize) * self._blocksize * (len(self._data) - 1) + len(self._loading_file(self._data[-1])))
        if self._augsize == 0:
            return total
        return total * self._augsize

    def __iter__(self):
        if self._cached_files:
            for batchXY in self._cached_files:
                yield batchXY
        else:
            scale = 0.05
            batchX, batchY = [], []
            for data_block in self.get_data():
                batchX.append(data_block[0])
                batchY.append(data_block[1])
                if len(batchX) == self._size:
                    batchXY = tc.tensor(np.vstack(batchX)), tc.tensor(np.vstack(batchY))
                    self._cached_files.append(batchXY); yield batchXY
                    batchX, batchY = [], []
                    if self._augsize > 0:
                        std = batchXY[0].std(axis=0)
                        for i in range(self._augsize):
                            noise = tc.randn(batchXY[0].shape) * std * scale
                            _batchXY = batchXY[0] + noise, batchXY[1]
                            self._cached_files.append(_batchXY); yield _batchXY
            if len(batchX) > 0:
                batchXY = tc.tensor(np.vstack(batchX)), tc.tensor(np.vstack(batchY))
                self._cached_files.append(batchXY); yield batchXY
                batchX, batchY = [], []
                if self._augsize > 0:
                    std = batchXY[0].std(axis=0)
                    for i in range(self._augsize):
                        noise = tc.randn(batchXY[0].shape) * std * scale
                        _batchXY = batchXY[0] + noise, batchXY[1]
                        self._cached_files.append(_batchXY); yield _batchXY

    def get_data(self, with_bar=False):
        self._pool.launch()
        self._pool.submit(self._data)
        pool = tqdm.tqdm(self._pool) if with_bar else self._pool
        for file in pool:
            for obj in file:
                yield obj

    def shuffle(self, seed=0):
        random.seed(seed)
        lastone = self._data.pop(-1)
        self._data.sort()
        random.shuffle(self._data)
        self._data.append(lastone)


def extract_layer_last_numpy(fpath, layerID):
    import pickle as pkl
    datasets = sorted([_ for _ in os.listdir(fpath) if _.startswith("group_") and _.endswith(".pt")])
    def loading_file(x):
        data = tc.load(fpath + '/' + x, weights_only=False)
        return [(_[0][layerID][-1], _[1]) for _ in data]
    temp = fpath + "_l%d_last" % layerID
    os.makedirs(temp, exist_ok=True)
    for each in tqdm.tqdm(datasets):
        opath = temp + '/' + each.replace(".pt", ".pkl")
        if True:#try:
            #if not os.path.exists(opath):
            pkl.dump(loading_file(each), open(opath, "wb"))
        #except Exception as e:
        #    print("Encountering Error at %s." % each)


if __name__ == "__main__":
    import sys
    for l in [int(sys.argv[1])]:
        print("Handling Layer %d" % l)
        #extract_layer_last_numpy("datasets/1round_ddxplus3k_mistral_train", l)
        extract_layer_last_numpy("datasets/dxy3_mistral_test", l)
        extract_layer_last_numpy("datasets/dxy3_mistral_train", l)
        #extract_layer_last_numpy("datasets/ddxplus1k_mistral_train", l)
        #extract_layer_last_numpy("datasets/ddxplus3k_mistral_train", l)
        

    


