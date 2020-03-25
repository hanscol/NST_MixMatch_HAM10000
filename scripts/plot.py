import matplotlib.pyplot as plt
import os
import numpy as np
import scipy

data = {'standard':{}, 'nst':{}, 'mix':{}, 'nstmix':{}}
results_dir = '/nfs/masi/hansencb/ham_nst_mixmatch/results'
nums = [779, 2335, 3888, 5442, 6998]
result_dirs = os.listdir(results_dir)
# result_dirs.sort()
for result_dir in os.listdir(results_dir):
    parts = result_dir.split('_')
    # key = '_'.join(parts[1:])

    fold = int(parts[0][-1])
    num = int(parts[1].split('d')[-1])
    nst = float(parts[2].split('a')[-1])
    lamb = float(parts[3].split('a')[-1])

    # for n in nums:
    #     if abs(num-n) < 5:
    #         num = n

    key = '{}_{}_{}'.format(num, nst, lamb)

    test_file = os.path.join(results_dir, result_dir, 'test_accuracy.txt')

    if os.path.isfile(test_file) and lamb < 101:
        with open(test_file, 'r') as f:
            acc = float(f.readline().strip())

        if nst == 0 and lamb == 0:
            if num not in data['standard']:
                data['standard'][num] = []
            data['standard'][num].append(acc)
        elif nst != 0 and lamb != 0:
            key = '{}_{}'.format(nst, lamb)
            if num not in data['nstmix']:
                data['nstmix'][num] = {}
            if key not in data['nstmix'][num]:
                data['nstmix'][num][key] = []
            data['nstmix'][num][key].append(acc)
        elif nst != 0:
            if num not in data['nst']:
                data['nst'][num] = {}
            if nst not in data['nst'][num]:
                data['nst'][num][nst] = []
            data['nst'][num][nst].append(acc)
        else:
            if num not in data['mix']:
                data['mix'][num] = {}
            if lamb not in data['mix'][num]:
                data['mix'][num][lamb] = []
            data['mix'][num][lamb].append(acc)

# nums = nums[1:-3]

img = []
for num in nums:
    d = data['nst'][num]
    nsts = list(d.keys())
    nsts.sort()

    row = []
    for nst in nsts:
        row.append(np.mean(d[nst]))
    img.append(row)

    data['nst'][num] = data['nst'][num][nsts[np.argmax(row)]]
    print('NST Numlabeled {} Best Lambda {}'.format(num, nsts[np.argmax(row)]))

img = np.array(img)

plt.figure()
plt.imshow(img)
plt.yticks(np.arange(0, len(nums)), nums)
plt.xticks(np.arange(0, len(nsts)), nsts)
plt.ylabel('Number of Labeled Subjects')
plt.xlabel('Nullspace Tuning Lambda')
plt.colorbar()
plt.draw()

img = []
for num in nums:
    d = data['mix'][num]
    lambs = list(d.keys())
    lambs.sort()

    row = []
    for lamb in lambs:
        row.append(np.mean(d[lamb]))
    img.append(row)
    data['mix'][num] = data['mix'][num][lambs[np.argmax(row)]]
    print('MixMatch Numlabeled {} Best Lambda {}'.format(num, lambs[np.argmax(row)]))

img = np.array(img)

plt.figure()
plt.imshow(img)
plt.yticks(np.arange(0, len(nums)), nums)
plt.xticks(np.arange(0, len(lambs)), lambs)
plt.ylabel('Number of Labeled Subjects')
plt.xlabel('MixMatch Tuning Lambda')
plt.colorbar()
plt.draw()


img = []
for num in nums:
    d = data['nstmix'][num]
    hypers = list(d.keys())

    row = []
    for hyper in hypers:
        if len(d[hyper])>1:
            row.append(np.mean(d[hyper]))
        else:
            row.append(0)
    img.append(row)
    data['nstmix'][num] = data['nstmix'][num][hypers[np.argmax(row)]]
    print('MixMatchNST Numlabeled {} Best Combo {}'.format(num, hypers[np.argmax(row)]))




keys = ['standard', 'mix', 'nst', 'nstmix']
legend = ['Baseline', 'Supervised', 'MixMatch', 'Nullspace Tuning', 'MixMatchNST']


auc_out_file = 'nlst_aucs.csv'

orig_auc = 81.5
x = np.arange(0, len(nums))
plt.figure()
plt.plot(x, np.zeros(len(nums)) + orig_auc)
# plt.fill_between(x, np.zeros(len(nums)) + orig_auc + 0.0211, np.zeros(len(nums)) + orig_auc -.0211, alpha=0.25)
plt.fill_between(x, np.zeros(len(nums)) + orig_auc + 0, np.zeros(len(nums)) + orig_auc -0, alpha=0.25)

with open(auc_out_file, 'w') as f:
    for i,k in enumerate(keys):
        y = []
        err = []
        f.write('{}\n'.format(legend[i+1]))
        for n in nums:
            f.write('{}\n'.format(','.join([str(x) for x in data[k][n]])))
            y.append(np.mean(data[k][n]))
            err.append(np.std(data[k][n])/np.sqrt(len(data[k][n])))

        y, err = np.array(y), np.array(err)
        plt.plot(x, y)
        plt.fill_between(x, y+err, y-err, alpha=0.25)

plt.legend(legend, loc='lower right')
plt.grid(True)
plt.xlabel('Number of Labeled Subjects')
plt.ylabel('Balanced Multiclass Accuracy')
plt.xticks(x, nums)
plt.draw()


plt.show()


