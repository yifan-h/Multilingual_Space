import os

langs = ["en", "af", "ar", "bg", "bn", "de", "el", "es", "et", "eu", 
         "fa", "fi", "fr", "he", "hi", "hu", "id", "it", "ja", "jv", 
         "ka", "kk", "ko", "ml", "mr", "ms", "my", "nl", "pt", "ru",
         "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "yo", "zh"]

def f1_cal(path):
    with open(path) as f:
        data = []
        for line in f:
            data.append(line)
    results = []
    for l in langs:
        for i in range(len(data)):
            d = data[i]
            if "Evaluation result  in " + l in d:
                r = data[i+1]
                f1 = round(int(r[r.find("f1 = "):][7:11])/100, 1)
                print(l, f1)
                results.append(f1)
    print("avg: ", sum(results)/len(results))

f1_cal("./results/panx_mbert_adapter.txt")
