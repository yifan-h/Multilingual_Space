import os

def f1_cal(path):
    with open(path) as f:
        data = []
        for line in f:
            data.append(line)
    results_em, results_f1 = [], []
    for i in range(len(data)):
        d = data[i]
        #l = data[i+1]
        if "- INFO - __main__ -   Results: {" in d:
            r = data[i]
            em = round(float(r[r.find("\'exact\': "):][9:14]), 1)
            f1 = round(float(r[r.find("\'f1\': "):][6:11]), 1)
            print(f1, "/", em)
            results_em.append(em)
            results_f1.append(f1)
    print("avg: ", sum(results_em)/len(results_em), sum(results_f1)/len(results_f1))

f1_cal("./results/mbert_xquad_5e-6_4x4.txt")
