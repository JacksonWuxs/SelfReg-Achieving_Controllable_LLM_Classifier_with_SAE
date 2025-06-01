import os
import random
from queue import Queue
from threading import Thread

import tqdm
import torch as tc


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
        while not self._closed:
            idx, args = self._inputs.get()
            if idx is None:
                self._inputs.put((None, None))
                break
            try:
                rslt = self._callfn(args)
            except Exception as e:
                rslt = e
            self._outputs.put((idx, rslt))

    def collect(self):
        return list(self)

    def submit(self, items):
        for idx, item in enumerate(items, self._size):
            self._inputs.put((idx, item))
            self._size += 1

    def launch(self):
        if len(self._pool) == 0 and self._closed is True:
            self._closed = False
            for _ in range(self._nworkers):
                self._pool.append(Thread(target=self._call, daemon=True))
                self._pool[-1].start()

    def close(self):
        if len(self._pool) > 0 and self._closed is False:
            self._closed = True
            self._inputs.put((None, None))

    def join(self):
        while len(self._pool) > 0:
            self._pool.pop(0).join()


class GroupActvDataset:
    def __init__(self, fpath, batch_size=1, workers=12, layerID=None):
        self._root = fpath
        self._size = batch_size
        self._layer = layerID
        self._data = [_ for _ in os.listdir(fpath) if _.startswith("group_") and _.endswith(".pt")]
        loading_file = lambda x: tc.load(fpath + '/' + x)
        self._pool = ThreadPool(loading_file, num_workers=workers, max_preload=workers*2)
        self._pool.launch()

    def __len__(self):
        return 640000
        size = 0
        for data in self.get_data(True):
            size += 1 #len(data)
        return size // self._size

    def __iter__(self):
        batch = []
        for data_block in self.get_data():
            if data_block.shape[0] != 0:
                #data_block = data_block.mean(axis=0, keepdims=True)
                data_block = data_block[-1:]
            while len(batch) + len(data_block) > self._size:
                skip = self._size - len(batch)
                batch.append(data_block[:skip])
                yield tc.vstack(batch); batch.clear();
                data_block = data_block[skip:]
            if len(data_block) > 0:
                batch.append(data_block)
        if len(batch) > 0:
            yield tc.vstack(batch); batch.clear();

    def get_data(self, with_bar=False):
        self._pool.submit(self._data)
        pool = tqdm.tqdm(self._pool) if with_bar else self._pool
        for file in pool:
            if not isinstance(file, list):
                print("Error Encounter: %s." % file)
                continue
            for obj in file:
                if self._layer is not None:
                    obj = obj[self._layer]
                if len(obj.shape) != 2:
                    obj = obj.unsqueeze(0)
                yield obj

    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self._data)
