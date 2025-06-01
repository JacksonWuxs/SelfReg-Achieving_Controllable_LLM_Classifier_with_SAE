import time
import re
import string
import json
import concurrent
import multiprocessing

import tqdm


from OpenaiAPI import Chatting



class URTIJudge:
    def __init__(self, key):
        instruct = "You are an experienced doctor.\n" +\
                   "* __Task:__ Check if the given feature and some of its example text spans show any symptomps related to " +\
                   "the diagnosis of Upper Respiratory Tract Infection (URTI), according to the Diagnosis Guideline.\n\n" +\
                   "You are given the feature with its name and several text spans from some patient-doctor conversations as examples. "+\
                   "Provide a short analysis of whether the feature connects with the diagnosis of URTI. " +\
                   "Note that, the duplicate text spans in the example are acceptable. " +\
                   "Please be as objective as possible. " +\
                   "Organize your final decision in the format of ``Final Decision: [[ Yes/Probably/Maybe/No ]]``." +\
                   "\n\n__Diagnosis Guideline__\n" +\
                   "* The feature does not need to satisfy all the following symptoms.\n" +\
                   "* The feature shows that the patient has a fever.\n" +\
                   "* The feature shows that the patient has nasal congestion, runny nose, sneezing, cough, phlegm, and/or hoarseness.\n" +\
                   "* The feature shows that the patient feels pains in their face, head, sore throat, ears, and/or eyes.\n" +\
                   "* The feature shows that the patient feels tired or sweating.\n" 
        self.model = Chatting.GPT4oMini(KEY, system=instruct, examples=None, cache=False,
                                   temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases):
        cases = map(self.format, cases)
        cases = self.model.batch_call(cases)
        return list(map(self.clean, cases))

    def format(self, case):
        case[1] = case[1].replace("\\n", "\n").replace("<s>[INST]", "").strip()
        #case[1] = "\nSpan".join(case[1].split("\nSpan")[:4])
        if 'cannot tell' in case[0].lower():
            return "Example Text Spans: \n%s" % case[1] 
        return "Feature Name: %s\nExample Text Spans: \n%s" % tuple(case)

    def clean(self, verify):
        temp = verify
        verify = verify[0].lower().split("decision")[-1]
        if "[[" in verify and "]]" in verify:
            verify = verify.split("[[", 1)[-1].rsplit("]]", 1)[0].strip()
        if verify.startswith(": "):
            verify = verify[2:].split(".", 1)[0].strip()
        if 'no' not in verify:
            print(temp)
        return verify + "|||" + temp[0]


class PneumoniaJudge:
    def __init__(self, key):                                                             
        instruct = "You are an experienced doctor.\n" +\
                   "* __Task:__ Check if the given feature and some of its example text spans show any symptomps related to " +\
                   "the diagnosis of Pneumonia, according to the Diagnosis Guideline.\n\n" +\
                   "You are given the feature with its name and several text spans from some patient-doctor conversations as examples. "+\
                   "Provide a short analysis of whether the feature connects with the diagnosis of Pneumonia. " +\
                   "Note that, the duplicate text spans in the example are acceptable. " +\
                   "Please be as objective as possible. " +\
                   "Organize your final decision in the format of ``Final Decision: [[ Yes/Probably/Maybe/No ]]``." +\
                   "\n\n__Diagnosis Guideline__\n" +\
                   "* The feature does not need to satisfy all the following symptoms.\n" +\
                   "* The feature shows that the patient has a fever.\n" +\
                   "* The feature shows that the patient has cough and/or phlegm.\n" +\
                   "* The feature shows that the patient feels pains in their chest or even shortness of breath.\n" +\
                   "* The feature shows that the patient feels tired or sweating.\n"
        self.model = Chatting.GPT4oMini(KEY, system=instruct, examples=None, cache=False, temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases):
        cases = map(self.format, cases)
        cases = self.model.batch_call(cases)
        return list(map(self.clean, cases))

    def format(self, case):
        case[1] = case[1].replace("\\n", "\n").replace("<s>[INST]", "").strip()
        #case[1] = "\nSpan".join(case[1].split("\nSpan")[:4])
        if 'cannot tell' in case[0].lower():
            return "Example Text Spans: \n%s" % case[1]
        return "Feature Name: %s\nExample Text Spans: \n%s" % tuple(case)

    def clean(self, verify):
        temp = verify
        verify = verify[0].lower().split("decision")[-1]
        if "[[" in verify and "]]" in verify:
            verify = verify.split("[[", 1)[-1].rsplit("]]", 1)[0].strip()
        if verify.startswith(": "):
            verify = verify[2:].split(".", 1)[0].strip()
        if 'no' not in verify:
            print(temp)
        return verify + "|||" + temp[0]
    

if __name__ == "__main__":
    KEY = "PUT_YOUR_OPENAI_KEY_HERE"
    
    model1 = URTIJudge(KEY)
    model2 = PneumoniaJudge(KEY)

    import sys
    file = sys.argv[1]
    print("Judging File: %s" % file)
    results = []
    with open(file, encoding="utf8") as f:
        headline = f.readline().strip().split("\t")
        assert headline == ["FeatureID", "Verify", "Summary", "Words"]
        for row in f.read().strip('\n').split("\n"):
            row = [_.replace("\\n", '\n').replace('\\t', '\t') for _ in row.split("\t")]
            results.append(['-'] + row)
            #if len(results) == 100: 
            #    break
    
    need_judge = [item for item in results if "span" in item[-1].lower()]
    relations1 = model1([_[3], _[4]] for _ in need_judge)
    relations2 = model2([_[3], _[4]] for _ in need_judge)
    for item, rela1, rela2 in zip(need_judge, relations1, relations2):
        item[0] = rela1 + "|||" + rela2
        

    from collections import Counter
    for i in [0, 2]:
        c = Counter([_[0].split("|||")[i] if _[0] != "-" else "-" for _ in results])
        print("Disease=%d: %.4f" % (i, (c["yes"] + c["probably"] + c["maybe"]) / len(results)))
        for cate, freq in c.items():
            print(cate, freq / len(results))
    with open(file.rsplit(".", 1)[0] + "_UnP.tsv", "w", encoding="utf8") as f:
        f.write("FeatureID\tTask\tVerify\tSummary\tWords\n")
        for task, idx, verify, summary, words in results:
            task = task.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            words = words.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            summary = summary.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            verify = verify.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            f.write("%s\t%s\t%s\t%s\t%s\n" % (idx, task, verify, summary, words))

