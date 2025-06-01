import time
import re
import string
import json
import math
import concurrent
import multiprocessing

import tqdm
import openai
import filelock


STORED_FILE = "./cache_myself.txt"


def synchronize(func, iters, batch_size=None, workers=None):
    if workers is None:
        workers = multiprocessing.cpu_count() * 2
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for batch in batchit(iters, batch_size):
            yield pool.map(func, batch)


def batchit(corpus, size):
    assert hasattr(corpus, "__iter__")
    assert size is None or isinstance(size, int) and size > 0
    batch = []
    for row in corpus:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch.clear()
    if len(batch) > 0:
        yield batch



class _APISetup:
    def __init__(self, secret_key, engine, function, do_cache=False, max_retry=None, cool_down=1.0):
        assert isinstance(secret_key, str)
        assert isinstance(engine, str)
        assert hasattr(openai, function)
        assert isinstance(max_retry, int) or max_retry is None
        assert isinstance(cool_down, (float, int)) and cool_down > 0.
        self._api = getattr(openai, function)
        self._key = openai.api_key = secret_key
        self._model = engine
        self._retry = max_retry
        self._cool = cool_down
        self._lock = filelock.FileLock(STORED_FILE + ".lock") if do_cache else None

    def __call__(self, *args, **kwrds):
        inputs = self.preprocess(*args, **kwrds)
        internals = self.create(**inputs)
        if self._lock:
            with self._lock:
                with open(STORED_FILE, "a+") as f:
                    store = {"INPUTS": inputs, "OUTPUTS": internals, "TIME": time.asctime()}
                    f.write(json.dumps(store) + '\n')
        return self.postprocess(internals)

    def batch_call(self, queries, batch_size=256, workers=None, bar=True):
        if bar:
            queries = list(queries)
            bar = tqdm.tqdm(total=math.ceil(len(queries) / batch_size))
        results = []
        for batch_result in synchronize(self.__call__, queries, 
                                        batch_size, workers):
            results.extend(batch_result)
            if bar:
                bar.update(1)
        return results
        
    def create(self, *args, **kwrds):
        tries = 0
        report = False
        while True:
            try:
                return self._api.create(model=self._model, *args, **kwrds)
            #except openai.error.Timeout:
            #    if tries == self._retry:
            #        print("Check you internet please!")
            #        return False
            except Exception as e:
                if not report:
                    print(("Unkown Error: %s" % e).replace("\n", "\\n"))
                    report = True
                    
            time.sleep(self._cool)
            tries += 1

    def preprocess(self, **kwrds):
        return kwrds

    def postprocess(self, outputs):
        return outputs



class Chatting(_APISetup):
    def __init__(self, secret_key, model, system=None, examples=None, cache=False, temperature=1.0, top_p=0.1, n=1, seed=42):
        _APISetup.__init__(self, secret_key, model, "ChatCompletion")
        self._params = {"temperature": temperature, "top_p": top_p, "n": n, "seed": seed}
        self.system = system
        self.examples = examples
        self._history = [] if cache else None

    @property
    def system(self):
        return self._instruct

    @system.setter
    def system(self, prompt):
        assert isinstance(prompt, str) or prompt is None
        self._instruct = []
        if prompt is not None:
            self._instruct.append({"role": "system", "content": prompt})

    @property
    def examples(self):
        return self._examples

    @examples.setter
    def examples(self, samples):
        if samples is None:
            samples = []
        if isinstance(samples, str):
            samples = [samples]
            
        new_examples = []
        for sample in samples:
            assert len(sample) == 2 and all(map(lambda _: isinstance(_, str), sample)), "each sample has two string terms."
            new_examples.append({"role": "system", "name": "example_user", "content": sample[0]})
            new_examples.append({"role": "system", "name": "example_assistant", "content": sample[1]})
        self._examples = new_examples

    def preprocess(self, new_query):
        new_query = {"role": "user", "content": new_query}
        inputs = self._instruct + self._examples
        if self._history is not None:
            inputs.extend(self._history)
            self._history.append(new_query)
        return {"messages": inputs + [new_query]} | self._params

    def postprocess(self, response):
        if response is False:
            if self._history:
                self._history.pop(-1)
            return False
        out_text = [_["message"]["content"] for _ in response["choices"]]
        if self._history:
            self._history.append({"role": "assistant", "content": out_text[0]})
        return out_text

    def clear_history(self):
        self._history.clear()

    def edit_history(self, text, idx=-1):
        assert len(self._history) > 0
        self._history[idx]["content"] = text

    @classmethod
    def GPT4oMini(cls, secret_key, model="gpt-4o-mini", system=None, examples=None, cache=False, temperature=0.0001, top_p=0.001, n=1):
        return cls(secret_key, model, system, examples, cache, temperature, top_p, n)




if __name__ == "__main__":
    KEY = "XXXXXXXXXXXXXX"
    model = Chatting.ChatGPT(KEY, model="gpt-4o-2024-08-06")
    while True:
        prompt = input("> User: ")
        print("> GPT: %s" % model(prompt)[0])
