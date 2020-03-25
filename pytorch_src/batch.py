from __future__ import print_function, division
import torch
import numpy as np
from pytorch_src.vat import VATLoss
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, lambda_u, rampup):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, rampup)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def train(model, device, loader, optimizer, config, ema_optimizer, epoch):
    model.train()

    correct = 0
    total_loss = 0

    with tqdm(total=len(loader)) as pbar:
        for batch_idx, sample in enumerate(loader):

            # if (batch_idx+1) * loader.batch_size < loader.dataset.stand_len:
            data = sample['image']
            target = sample['target']
            target = torch.zeros(target.shape[0], 7).scatter_(1, target.view(-1, 1), 1)
            data, target = data.to(device), target.to(device)

            ul_data1 = sample['ul_img1']
            ul_data2 = sample['ul_img2']
            ule_data1 = sample['ule_img1']
            ule_data2 = sample['ule_img2']

            ul_data1, ul_data2 = ul_data1.to(device), ul_data2.to(device)
            ule_data1, ule_data2 = ule_data1.to(device), ule_data2.to(device)

            if config.mixmatch:
                with torch.no_grad():
                    outputs_ul1 = model(ul_data1)
                    outputs_ul2 = model(ul_data2)
                    p = (torch.softmax(outputs_ul1, dim=1) + torch.softmax(outputs_ul2, dim=1)) / 2
                    pt = p ** (1 / config.T)
                    targets_u = pt / pt.sum(dim=1, keepdim=True)

                    targets_u = targets_u.detach()

                #mixup
                all_inputs = torch.cat([data, ul_data1, ul_data2], dim=0)
                all_targets = torch.cat([target, targets_u, targets_u], dim=0)

                l = np.random.beta(config.alpha, config.alpha)

                l = max(l, 1 - l)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
                mixed_input = list(torch.split(mixed_input, config.batch_size))
                mixed_input = interleave(mixed_input, config.batch_size)

                logits = [model(mixed_input[0])]
                for input in mixed_input[1:]:
                    logits.append(model(input))

                # put interleaved samples back
                logits = interleave(logits, config.batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)

                Lx, Lu, w = SemiLoss()(logits_x, mixed_target[:config.batch_size], logits_u, mixed_target[config.batch_size:],
                                      epoch + batch_idx / len(loader), config.lambda_u, config.rampup)

                loss = Lx + w * Lu

            else:
                output = model(data)
                loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1) * target, dim=1))

            if config.null_space_tuning:
                # compute guessed labels of unlabel samples
                outputs_ub = model(ul_data1)
                outputs_ub2 = model(ul_data2)
                outputs_e = model(ule_data1)
                outputs_e2 = model(ule_data2)

                pu = (torch.softmax(outputs_ub, dim=1) + torch.softmax(outputs_ub2, dim=1)) / 2
                ptu = pu ** (1 / config.T)
                targets_u1 = ptu / ptu.sum(dim=1, keepdim=True)

                pe = (torch.softmax(outputs_e, dim=1) + torch.softmax(outputs_e2, dim=1)) / 2
                pte = pe ** (1 / config.T)
                targets_e = pte / pte.sum(dim=1, keepdim=True)

                nst_loss = torch.mean((targets_u1 - targets_e) ** 2)
                loss += linear_rampup(epoch, config.rampup) * config.alpha * nst_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            total_loss += loss.item()

            pbar.set_description('Epoch {}\tAvg Loss: {:.4f}'.format(epoch, total_loss/(batch_idx+1)))
            pbar.update(1)

    avg_loss = total_loss / (batch_idx+1)

    return avg_loss


def test(model, device, loader):
    model.eval()

    correct = 0
    total_loss = 0
    total_evaluated = 0

    init = True
    confusion = None
    pred_all = torch.tensor([]).type(torch.LongTensor)
    target_all = torch.tensor([]).type(torch.LongTensor)

    with open('../scores4.csv', 'w') as f:
        f.write('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for batch_idx, sample in enumerate(loader):
                    data = sample['image']
                    target = sample['target']
                    file = sample['file']

                    labels = target.to(device)
                    target = torch.zeros(target.shape[0], 7).scatter_(1, target.view(-1, 1), 1)
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    output = output.to(device)

                    if init:
                        confusion = np.zeros((output.shape[1],output.shape[1]))
                        init = False

                    loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1) * target, dim=1))

                    pred = output.max(1, keepdim=True)[1]

                    pred_all = torch.cat((pred_all, pred.cpu()))
                    target_all = torch.cat((target_all, labels.cpu()))
                    correct_mask = pred.eq(labels.view(-1,1))
                    correct += correct_mask.sum().item()

                    for i in range(len(correct_mask)):
                        confusion[int(pred[i]), int(labels[i])] += 1

                    total_evaluated += labels.shape[0]
                    total_loss += loss.item()
                    avg_loss = total_loss / (batch_idx+1)
                    accuracy =  100 * balanced_accuracy_score(target_all, pred_all)
                    pbar.set_description('       \tAvg Loss: {:.4f}  Accuracy: {:.2f}%'.format(avg_loss, accuracy))
                    pbar.update(1)

                    scores = F.softmax(output, dim=1).cpu()
                    for i in range(scores.shape[0]):
                        id = file[i].split('/')[-1].split('.')[0]
                        f.write(id+','+','.join([str(x.item()) for x in scores[i]])+'\n')


    avg_loss = total_loss / (batch_idx+1)
    accuracy = 100 * balanced_accuracy_score(target_all, pred_all)

    return avg_loss, accuracy, confusion
