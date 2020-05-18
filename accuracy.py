#!/usr/bin/env python3

import sys
import os

import numpy as np

from task_config import yle_labels, ylilauta_labels


def main(argv):
    if len(argv) != 4:
        print('Usage: {} TASK GOLD PRED'.format(os.path.basename(__file__)))
        return 1
    task, gold_fn, pred_fn = argv[1:]

    labels_by_task = {
        'yle': yle_labels,
        'ylilauta': ylilauta_labels,
    }
    if task.lower() not in labels_by_task:
        raise ValueError('Unknown task {}'.format(task))
    labels = labels_by_task[task.lower()]
    label_map = { i: l for i, l in enumerate(labels) }

    gold = []
    with open(gold_fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            label, text = l.split(None, 1)
            assert label.startswith('__label__'), 'no label'
            label = label[len('__label__'):]
            if label not in labels:
                raise ValueError('Unknown label {}'.format(label))
            gold.append(label)

    pred = []
    with open(pred_fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split()
            if len(fields) != len(labels):
                raise ValueError('Unexpected number of probabilities')
            probs = [float(i) for i in fields]
            p = np.argmax(probs)
            pred.append(label_map[p])

    if len(gold) != len(pred):
        raise ValueError('gold/pred number mismatch')

    correct, total = 0, 0
    for g, p, in zip(gold, pred):
        if g == p:
            correct += 1
        total += 1

    print('{:.2%}'.format(correct/total))

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
