import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import copy


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
            Sampler(dataset, idxs),
            batch_size=args.bs, 
            shuffle=True
        )
        
        loss_function = nn.CrossEntropyLoss().to(device)

        losses = []

        for epoch in range(args.epoch):
            batch_losses = []
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

                if not idx % 10 and args.verbose:
                    print(f"\t\tCommunication round {round+1} | Device {device_num+1} | Epoch {epoch+1} | Batch {idx+1} | Training loss: {loss.item()}\n")

                batch_losses.append(loss.item())

            avg_batch_loss = sum(batch_losses)/len(batch_losses)
            losses.append(avg_batch_loss)

            print(f"\tCommunication round {round+1} | Device {device_num+1} | Epoch {epoch+1} | Training loss: {avg_batch_loss}")

            # Learning rate decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * args.lr_decay

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

                if not idx % 10 and args.verbose:
                    print(f"\t\tCommunication round {round+1} | Batch {idx} | Testing loss: {loss.item()} | Testing accuracy: {batch_accuracy}")

        return correct/total, sum(losses)/len(losses)
