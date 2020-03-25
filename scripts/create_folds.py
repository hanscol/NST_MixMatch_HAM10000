import os
import random
import math

def write_csv(data, fname):
    with open(fname, 'w') as f:
        for sample in data:
            f.write('{},{}\n'.format(sample[0], sample[1]))

data_dir = '/scratch/hansencb/Classification_datasets/SkinLesions'
class_dirs = ['0', '1', '2', '3', '4', '5', '6']
out_dir = 'folds'
k = 5
val_percent = 0.025
os.makedirs(out_dir, exist_ok=True)

data_by_class = {}
total = 0
for c in class_dirs:
    label = int(c)
    data_by_class[label] = []
    for img_file in os.listdir(os.path.join(data_dir,c)):
        path = os.path.join(data_dir, c, img_file)
        data_by_class[label].append((path,label))
    total += len(data_by_class[label])
    random.shuffle(data_by_class[label])

val_num = int(val_percent * total / len(data_by_class))
percent_labeled = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

for i in range(k):
    fold_dir = os.path.join(out_dir, 'fold{}'.format(i))
    os.makedirs(fold_dir, exist_ok=True)

    test = []
    train = {}
    val = []
    for c in data_by_class:
        train[c] = []
        fold_num = int(len(data_by_class[c])/k)
        test.extend(data_by_class[c][fold_num*i:fold_num*(i+1)])
        train[c].extend(data_by_class[c][0:fold_num*i])
        train[c].extend(data_by_class[c][fold_num*(i+1):])

        for j in range(val_num):
            idx = random.randint(0, len(train[c])-1)
            val.append(train[c][idx])
            del train[c][idx]

    write_csv(test, os.path.join(fold_dir, 'test.csv'))
    write_csv(val, os.path.join(fold_dir, 'val.csv'))

    for pl in percent_labeled:
        labeled = []
        unlabeled = []
        for c in data_by_class:
            numl = math.ceil(pl*len(train[c]))
            unlabeled.extend(train[c][numl:])
            labeled.extend(train[c][0:numl])
        write_csv(labeled, os.path.join(fold_dir, 'train_labeled_{}.csv'.format(len(labeled))))
        write_csv(unlabeled, os.path.join(fold_dir, 'train_unlabeled_{}.csv'.format(len(labeled))))







