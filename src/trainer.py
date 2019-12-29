import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import copy
import math
from pycm import *

"""
Dataset for training devices
""" 
class Sampler(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

    def __len__(self):
        return len(self.idxs)


class Trainer():
    def train(self, dataset, idxs, round, device_num, device, model, args):
        model.to(device)
        model.train()   # Train mode

        if args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr, 
                momentum=args.sgd_momentum
            )
        elif args.optim == "adagrad":
            optimizer = torch.optim.Adagrad(
                model.parameters(), 
                lr=args.lr
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=args.lr
            )

        dataloader = DataLoader(
            Sampler(dataset, idxs) if args.learning == "f" else dataset,
            batch_size=args.bs, 
            shuffle=True
        )
        
        loss_function = nn.CrossEntropyLoss().to(device)

        losses = []

        epoch = args.epoch if args.learning == "f" else 1

        for epoch in range(epoch):
            batch_losses = []
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

                if args.learning == "f":
                    if args.verbose:
                        print(f"\t\tCommunication round {round+1} | Device {device_num+1} | Epoch {epoch+1} | Batch {idx+1} | Training loss: {loss.item()}\n")
                else:
                    if not idx % 10:
                        print(f"\tEpoch {round+1} | {idx*args.bs}/{len(dataset)} | Training loss: {loss.item()}")

                batch_losses.append(loss.item())

            avg_batch_loss = sum(batch_losses)/len(batch_losses)
            losses.append(avg_batch_loss)

            if args.learning == "f":
                print(f"\tCommunication round {round+1} | Device {device_num+1} | Epoch {epoch+1} | Training loss: {avg_batch_loss}")
            else:
                print(f"\nEpoch {round+1} | Average training loss: {avg_batch_loss}\n")

        return copy.deepcopy(model.state_dict()), sum(losses)/len(losses)


class Tester():
    def test(self, dataset, round, device, model, args):
        model.to(device)
        model.eval()    # Evaluate mode

        dataloader = DataLoader(
            dataset,
            batch_size=args.bs, 
            shuffle=False
        )
        loss_function = nn.CrossEntropyLoss().to(device)

        losses = []
        total, correct = 0, 0

        predicts = []
        targets = []

        with torch.no_grad():
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_function(output, target)
                losses.append(loss.item())

                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                batch_accuracy = (predicted == target).sum().item()/target.size(0)

                # For calculating metrics
                predicts += predicted.tolist()
                targets += target.tolist()

                if not idx % 10 and args.verbose:
                    if args.learning == "f":
                        print(f"\t\tCommunication round {round+1} | Batch {idx} | Testing loss: {loss.item()} | Testing accuracy: {batch_accuracy}")
                    else:
                        print(f"\t\tEpoch {round+1} | Batch {idx} | Testing loss: {loss.item()} | Testing accuracy: {batch_accuracy}")

        # Print metrics
        cm = ConfusionMatrix(actual_vector=targets, predict_vector=predicts)
        
        if args.verbose:
            print(cm)

        auc = list(cm.AUC.values())

        return correct/total, sum(losses)/len(losses), sum(auc)/len(auc), cm.Kappa