import time
import re
import string
import json
import concurrent
import multiprocessing

import tqdm


from OpenaiAPI import Chatting


class TextSpanExplainer:
    def __init__(self, key):
        instruct = "You are studying a neural network. Each neuron looks for one particular concept/topic/theme/behavior/pattern. " +\
               "Look at some text spans the neuron activates for and guess what the neuron is looking for. " +\
               "Note that, some neurons may only look for a particular text pattern, while some others may be interestedin very abstractive concepts. " +\
               "Pay more attention to the end of each text span as they supposed to be more correlated to the neuron behavior. " +\
               "Don't list examples of text spans and keep your summary as detail as possible. " +\
               "If you cannot summarize most of the text spans, you should say ``Cannot Tell.``"
        examples = [("""Span 1: w.youtube.com/watch?v=5qap5aO4z9A
Span 2: youtube.come/yegfnfE7vgDI
Span 3: {'token': 'bjXRewasE36ivPBx
Span 4: /2023/fid?=0gBcWbxPi8uC""", 
                     "Base64 encoding for web development."),
                     ("""Span 1: cross-function\\n
Span 2: cross-function.
Span 3: this is a cross-function
Span 4: Cross-Function""",
                      'Particular text pattern "cross-function".'),
                     ("""Span 1: novel spectroscopic imaging platform
Span 2: and protein evolutionary network modeling
Span 3: reactions-centric biochemical model
Span 4: chaperone interaction network""",
                      "Biological terms."),
                     ("""Span 1: is -17a967
Span 2: what is 8b8 - 10ad2
Span 3: 83 -11111011001000001011
Span 4: is -c1290 - -1""",
                      "Synthetic math: Arithmetic, numbers with small digits, in unusual bases."),
                     ("""Span 1: Could you please provide me some
Span 2: USER__: Could you please provide me some
Span 3: USER__: Could you please provide me some
Span 4: Sure! Could you please provide me some""",
                      'Particular text pattern "Could you please provide me some"')]
        self.model = Chatting.GPT4oMini(KEY, cache=False,
                                   system=instruct, examples=examples,
                                   temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases):
        if isinstance(cases, str):
            cases = [cases]
        if not isinstance(cases, (tuple, list)):
            cases = list(cases)
        cases = map(self.format, cases)
        return list(map(self.clean, self.model.batch_call(cases)))

    def format(self, raw):
        raw = raw.replace("\\n", "\n").replace("<s>[INST]", "").strip()
        return raw
        return "\nSpan".join(raw.split("\nSpan")[:4])

    def clean(self, summaries):
        temp = set(_ for _ in summaries if "cannot tell" not in _.lower())
        if len(temp) == 0:
            return "Cannot Tell."
        return " or ".join(_.split(".")[0] for _ in temp) + '.'


class TextSpanJudge:
    def __init__(self, key):
        instruct = "You are an linguistic expert. " +\
                   "Provide a short analysis on whether the text spans well represent the given concept/topic/theme/pattern. " +\
                   "Note that, the text spans share the same phrases or are duplicated are acceptable. " +\
                   "Please do not be too mean but be as subjective as possible. " +\
                   "Organize your final decision in the format of ``Final Decision: [[ Yes/Probably/Maybe/No ]]``."
        self.model = Chatting.GPT4oMini(KEY, system=instruct, examples=None, cache=False,
                                   temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases):
        cases = map(self.format, cases)
        cases = self.model.batch_call(cases)
        return list(map(self.clean, cases))

    def format(self, case):
        case[1] = case[1].replace("\\n", "\n").replace("<s>[INST]", "").strip()
        #case[1] = "\nSpan".join(case[1].split("\nSpan")[:4])
        return "Concept/Topic/Theme/Pattern: %s.\nWords: %s" % tuple(case)

    def clean(self, verify):
        temp = verify
        verify = verify[0].lower().split("decision")[-1]
        if "[[" in verify and "]]" in verify:
            verify = verify.split("[[", 1)[-1].rsplit("]]", 1)[0].strip()
        if verify.startswith(": "):
            verify = verify[2:].split(".", 1)[0].strip()
        return verify
    

if __name__ == "__main__":
    KEY = "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" 
    
    model = TextSpanExplainer(KEY)
    judge = TextSpanJudge(KEY)

    import sys
    file = sys.argv[1]
    print("Annotating File: %s" % file)
    with open(file, encoding="utf8") as f:
        headline = f.readline().strip().split("\t")
        idx = headline.index("FeatureID")
        text = headline.index("Words")
        fullset = [(_.split("\t")[idx], _.split("\t")[text].strip()) for _ in f.read().strip('\n').split("\n")]
    results = [[x[0], "-", "Cannot Tell.", x[1]] for x in fullset]#[:100]
    
    need_explain = [item for item in results if len(item[3]) > 0]
    explanations = model(_[3] for _ in need_explain)
    for item, expl in zip(need_explain, explanations):
        item[1], item[2] = 'no', expl
        
    need_verify = [item for item in results if "cannot tell" not in item[2].lower()]
    verifications = judge([_[2], _[3]] for _ in need_verify)
    for item, verify in zip(need_verify, verifications):
        item[1] = verify


    from collections import Counter
    c = Counter([_[1] for _ in results])
    print("Explainability: %.4f" % ((c["yes"] + c["probably"]) / len(results)))
    for cate, freq in c.items():
        print(cate, freq / len(results))
    with open(file.rsplit(".", 1)[0] + "_explained.tsv", "w", encoding="utf8") as f:
        f.write("FeatureID\tVerify\tSummary\tWords\n")
        for idx, verify, summary, words in results:
            words = words.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            summary = summary.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            verify = verify.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            f.write("%s\t%s\t%s\t%s\n" % (idx, verify, summary, words))

