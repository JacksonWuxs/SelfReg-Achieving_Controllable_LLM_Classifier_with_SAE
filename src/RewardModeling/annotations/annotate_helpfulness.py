import time
import re
import string
import json
import concurrent
import multiprocessing

import tqdm


from OpenaiAPI import Chatting



class HelpfulJudge:
    def __init__(self, key):
        instruct = "You are a linguistic expert. " +\
                   "__Your job is to identify some text-based features whether they can be used to judge the helpfulness of a chatbot, according to the following Guideline of Helpfulness.__ " +\
                   "You are given the feature with its name and several text spans from a user-chatbot conversation as examples. " +\
                   "Provide a short analysis on whether the feature could be used to judge the helpfulness of the chatbot. " +\
                   "Note that, the duplicated text spans in the example are acceptable. " +\
                   "Please do not be too mean but be as subjective as possible. " +\
                   "Organize your final decision in the format of ``Final Decision: [[ Yes/Probably/Maybe/No ]]``. " +\
                   "\n\n__Guideline of Helpfulness__\n" +\
                   "* The feature does not need to satisfy with all the following rules.\n" +\
                   "* The feature shows that the chatbot can write in clear and polite language.\n" +\
                   "* The feature shows that the chatbot can answer a challenging question from a user (e.g., programming, reasoning, solving math problems).\n" +\
                   '* The feature shows that the chatbot is sensitive to internationality (e.g., "football" should not mean American football, and "the president" does not necessarily mean the US president).\n' +\
                   '* The feature shows that the chatbot knows to ask for clarification if a user question is too confusing.\n' +\
                   '* The feature shows that the chatbot will not give overly long or rambling answers.\n' +\
                   '* The feature shows that the chatbot is knowledgeable in world-wide knowledge.\n' +\
                   '* The feature should not come from the user query.'

        self.model = Chatting.GPT4oMini(KEY, system=instruct, examples=None, cache=False,
                                   temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases):
        cases = map(self.format, cases)
        cases = self.model.batch_call(cases)
        return list(map(self.clean, cases))

    def format(self, case):
        case[1] = case[1].replace("\\n", "\n").replace("<s>[INST]", "").strip()
        case[1] = "\nSpan".join(case[1].split("\nSpan")[:4])
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
    KEY = "PUT YOUR OPENAI KEY HERE"
    
    model = HelpfulJudge(KEY)

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
    
    need_judge = [item for item in results if 'span' in item[4].lower() and 'cannot tell' not in item[3]]
    relations = model([_[3], _[4]] for _ in need_judge)
    for item, rela in zip(need_judge, relations):
        item[0] = rela
        

    from collections import Counter
    c = Counter([_[0].split("|||")[0] for _ in results])
    print("Explainability: %.4f" % ((c["yes"] + c["probably"]) / len(results)))
    for cate, freq in c.items():
        print(cate, freq / len(results))
    with open(file.rsplit(".", 1)[0] + "_TaskMiniReward.tsv", "w", encoding="utf8") as f:
        f.write("FeatureID\tTask\tVerify\tSummary\tWords\n")
        for task, idx, verify, summary, words in results:
            task = task.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            words = words.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            summary = summary.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            verify = verify.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            f.write("%s\t%s\t%s\t%s\t%s\n" % (idx, task, verify, summary, words))

