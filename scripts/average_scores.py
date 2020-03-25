import os
import numpy as np

score_file = 'scores{}.csv'
out_score_file = 'scores.csv'
avg_scores = np.zeros((1512,7))
ids = []
first_row = ''
for i in range(5):
    scores = []
    with open(score_file.format(i), 'r') as f:
        first_row = f.readline()
        for line in f.readlines():
            scores.append([float(x) for x in line.strip().split(',')[1:]])
            if i == 0:
                ids.append(line.split(',')[0])
    scores = np.array(scores)
    avg_scores = avg_scores + scores

avg_scores /= 5
with open(out_score_file, 'w') as f:
    f.write(first_row)
    for i, id in enumerate(ids):
        f.write(id +','+','.join([str(x) for x in avg_scores[i]])+'\n')