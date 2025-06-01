import os
import sys

import numpy as np
import scipy.stats as stats

def extract(fpath):
    with open(fpath, encoding='utf8') as f:
        for row in f:
            if row.startswith("Best"):
                break
        else:
            return {}
    scores = {"Seed": int(fpath.split("seed")[1].split(".")[0])}
    for each in row[4:].split(" | "):
        key, val = each.split("=")
        scores[key] = float(val)
    return scores



def summary_scores(folder):
    total_scores = {}
    for fpath in os.listdir(folder):
        if fpath.endswith(".log"):
            for key, val in extract(folder + '/' + fpath).items():
                if key not in total_scores:
                    total_scores[key] = []
                total_scores[key].append(val)

    print("\n")
    N = max(map(len, total_scores.values()))
    print("Mean Folder=%s | Number=%d | %s" % (folder, N, " | ".join(key + '=%.4f' % np.mean(vals) for key, vals in total_scores.items())))
    print("Std Folder=%s | Number=%d | %s" % (folder, N, " | ".join(key + '=%.4f' % np.std(vals) for key, vals in total_scores.items())))
    return total_scores



if __name__ == "__main__":
    score1 = summary_scores(sys.argv[1])
    if len(sys.argv) > 2:
        score2 = summary_scores(sys.argv[2])
        print("\n\nComparision\n")
        for key in score1:
            s1, s2 = score1[key], score2[key]
            s1, s2 = [], []
            for seed in score1["Seed"]:
                s1.append(score1[key][score1["Seed"].index(seed)])
                s2.append(score2[key][score2["Seed"].index(seed)])
            t_stat, p_val = stats.ttest_rel(s1, s2, alternative="greater")
            #Good one: t_stat, p_val = stats.ttest_ind(s1, s2, equal_var=True, alternative="greater")
            print("%s | avg1=%.4f | std1=%.4f | avg2=%.4f | std2=%.4f | t-stat=%.4f | p-value=%.4f" % (
                key, np.mean(s1), np.std(s1), np.mean(s2), np.std(s2), t_stat, p_val))
