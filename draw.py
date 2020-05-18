#!/usr/bin/env python3

import sys
import re
import numpy as np
import matplotlib.pyplot as plt

datapoint_re = re.compile(r'^.*init_checkpoint\s(.*)\sdata_dir\s(ylilauta|yle|ylilauta\/trunc|yle\/trunc)\/?-?(\d+)-percent.*mean\s([\d\.]+).*$')

def plot(x, y, fname):
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    x_max = max(x)+10
    y_max = max(y)+1
    plt.axis([0, x_max, 0, y_max])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(fname)

def parse_result_file(filename):
    results = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            m = re.match(datapoint_re, line)
            if not m:
                continue
            model = m.group(1)
            data = m.group(2)
            percent = m.group(3)
            mean = m.group(4)
            results.append((model, data, percent, mean))
    return results

def parse_file(filename):
    x=[]
    y=[]
    with open(filename) as f:
        while True:
            line=f.readline()
            if not line:
                break
            step, loss=line.split('\t')
            x.append(int(step))
            y.append(float(loss))
    return x, y


if __name__ == '__main__':
    assert len(sys.argv)==3, 'First arg: file containing steps and losses, Second arg: output pic file name'
    # first arg file containing steps and losses
    #/scratch/project_2002085/lihsin/bilingual-bert/jointvoc/logs/delme_losses.txt
    filename = sys.argv[1]
    results = parse_result_file(filename)
    corpora = list(set([data for _, data, _, _ in results]))
    models = list(set([model for model, _, _, _ in results]))

    for corpus in corpora:
        corpus_name = '-'.join(corpus.split('/'))
        for model in models:
            datapoints = [(int(percentage), float(mean)) for m, data, percentage, mean in results if data==corpus if m==model]
            datapoints.sort(key=lambda x:x[0])
            x = np.log10([x for x, _ in datapoints])
            y = np.array([y for _, y in datapoints])
            #print(datapoints)
            plt.plot(x, y, label=model)
        plt.xlabel('Training data size')
        plt.ylabel('Accuracy')
        plt.title(corpus_name)
        plt.legend()
        plt.savefig(sys.argv[2]+corpus_name)
        plt.clf()
        


