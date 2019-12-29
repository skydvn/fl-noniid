import time
from arg_parser import Parser
from utils import Utils
import torch
import numpy as np
import random
from tqdm import tqdm
from trainer import Trainer, Tester
import copy
from models import *
from torch.utils.data import Dataset, DataLoader


class FederatedLearning():
    def __init__(self, args):
        self.args = args

    def run(self):
        start = time.time()

        # Print arguments
        if self.args.verbose:
            print("Arguments:")
            print(f"\t{self.args}")

        # Set training on CPU/GPU
        device = "cpu"
        if self.args.gpu is not None:
            if torch.cuda.is_available():
                device = "cuda"
            torch.cuda.set_device(self.args.gpu)

        # Set manual random seed
        if not self.args.random_seed:
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            torch.backends.cudnn.deterministic = True

        utils = Utils()

        # Get dataset and data distribution over devices
        train_dataset, test_dataset, device_idxs = utils.get_dataset_dist(self.args)

        # Get number of classes (MNIST: 10, CIFAR10: 10)
        if self.args.dataset == "mnist":
            num_classes = 10
        else:
            num_classes = 10
        
        # Set training model (VGG11/LeNet/ResNet18)
        if self.args.model == "vgg11":
            model = VGG("VGG11", num_classes)
        elif self.args.model == "lenet":
            model = LeNet(num_classes)
        else:
            model = ResNet18(num_classes)

        # Optimization technique
        if self.args.warmup_model:
            weights = utils.warmup_model(model, train_dataset, test_dataset, device, self.args)
            model.load_state_dict(weights)

        avg_weights_diff = []
        global_train_losses = []
        global_test_losses = []
        global_accuracies = []
        global_aucs = []
        global_kappas = []


        for round in tqdm(range(self.args.round)):
            # Train step
            print(f"\nRound {round+1} Training:")

            local_weights = []
            local_losses = []

            # Select fraction of devices (minimum 1 device)
            train_devices = random.sample(
                range(self.args.num_devices), 
                max(1, int(self.args.num_devices*self.args.frac))
            )

            print(f"\tDevices selected: {[x+1 for x in train_devices]}\n")

            # Train on each device and return weights and loss
            for device_num in train_devices:
                weights, loss = Trainer().train(
                    train_dataset, 
                    device_idxs[device_num], 
                    round, 
                    device_num, 
                    device, 
                    copy.deepcopy(model),   # Avoid continuously training same model on different devices
                    self.args
                )

                local_weights.append(weights)
                local_losses.append(loss)

            avg_weights = utils.fed_avg(local_weights) # Federated averaging

            model.load_state_dict(avg_weights)  # Load new weights

            if self.args.cal_para_diff:
                avg_weight_diff = utils.cal_avg_weight_diff(local_weights, avg_weights)
            else:
                 avg_weight_diff = 0
            avg_loss = sum(local_losses)/len(local_losses)

            print(f"\n\tRound {round+1} | Average weight difference: {avg_weight_diff}")
            print(f"\tRound {round+1} | Average training loss: {avg_loss}\n")

            global_train_losses.append(avg_loss)
            avg_weights_diff.append(avg_weight_diff)



            # Test step
            print(f"Round {round+1}  Testing:")
            accuracy, loss, auc, kappa = Tester().test(
                test_dataset, 
                round, 
                device, 
                copy.deepcopy(model), 
                self.args
            )

            print(f"\tRound {round+1} | Average accuracy: {accuracy}")
            print(f"\tRound {round+1} | Average testing loss: {loss}\n")
            print(f"\tRound {round+1} | Average AUC: {auc}")
            print(f"\tRound {round+1} | Kappa: {kappa}\n")

            global_test_losses.append(loss)
            global_accuracies.append(accuracy)
            global_aucs.append(auc)
            global_kappas.append(kappa)


            # Quit early if satisfy certain situations
            if accuracy >= self.args.train_until_acc/100:
                print(
                    f"Accuracy reached {self.args.train_until_acc/100} in round {round+1}, stopping...")
                break
            if self.args.stop_if_improvment_lt:
                if round > 0:
                    if global_accuracies[-2]+self.args.stop_if_improvment_lt/100 >= global_accuracies[-1]:
                        break

        end = time.time()
        print(f"\nTime used: {time.strftime('%H:%M:%S', time.gmtime(end-start))}")

        # Print final results
        print("\nAverage weight differences:")
        print(f"\t{avg_weights_diff}\n")
        print("Losses on training data:")
        print(f"\t{global_train_losses}\n")
        print("Losses on testing data:")
        print(f"\t{global_test_losses}\n")
        print("Accuracies on testing data:")
        print(f"\t{global_accuracies}\n")
        print("Average AUCs on testing data:")
        print(f"\t{global_aucs}\n")
        print("Kappas on testing data:")
        print(f"\t{global_kappas}\n")

        print(f"Final accuracy: {global_accuracies[-1]}")
        print(f"Final loss: {global_test_losses[-1]}\n")

        # Write results to file
        if self.args.save_results:
            utils.save_results_to_file(
                self.args, 
                avg_weights_diff,
                global_train_losses, 
                global_test_losses, 
                global_accuracies,
                global_aucs,
                global_kappas
            )


class CentralizedLearning():
    def __init__(self, args):
        self.args = args

    def run(self):
        start = time.time()

        # Print arguments
        if self.args.verbose:
            print("Arguments:")
            print(f"\t{self.args}")

        # Set training on CPU/GPU
        device = "cpu"
        if self.args.gpu is not None:
            if torch.cuda.is_available():
                device = "cuda"
            torch.cuda.set_device(self.args.gpu)

        # Set manual random seed
        if not self.args.random_seed:
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            torch.backends.cudnn.deterministic = True

        utils = Utils()

        # Get dataset and data distribution over devices
        train_dataset, test_dataset, _ = utils.get_dataset_dist(self.args)

        # Get number of classes (MNIST: 10, CIFAR100: 10)
        if self.args.dataset == "mnist":
            num_classes = 10
        else:
            num_classes = 10
        
        # Set training model (VGG11/LeNet/ResNet18)
        if self.args.model == "vgg11":
            model = VGG("VGG11", num_classes)
        elif self.args.model == "lenet":
            model = LeNet(num_classes)
        else:
            model = ResNet18(num_classes)

        train_losses = []
        test_losses = []
        accuracies = []
        aucs = []
        kappas = []

        if self.args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.lr, 
                momentum=self.args.sgd_momentum
            )
        elif self.args.optim == "adagrad":
            optimizer = torch.optim.Adagrad(
                model.parameters(), 
                lr=self.args.lr
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.args.lr
            )

        for epoch in tqdm(range(self.args.epoch)):
            # Train step
            print(f"Epoch {epoch+1} Training:")

            """
            model.to(device)
            model.train()   # Train mode

            dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.bs, 
                shuffle=True
            )
            
            loss_function = nn.CrossEntropyLoss().to(device)

            batch_losses = []
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

                if not idx % 10:
                    print(f"\tEpoch {epoch+1} | {idx*self.args.bs}/{len(train_dataset)} | Training loss: {loss.item()}")

                batch_losses.append(loss.item())

            train_losses.append(sum(batch_losses)/len(batch_losses))

            print(f"\nEpoch {epoch+1} | Average training loss: {loss}\n")
            """

            weights, loss = Trainer().train(
                train_dataset, 
                0, 
                epoch, 
                0, 
                device, 
                copy.deepcopy(model),
                self.args
            )

            model.load_state_dict(weights)
            train_losses.append(loss)



            # Test step
            print(f"Epoch {epoch+1}  Testing:")
            accuracy, loss, auc, kappa = Tester().test(
                test_dataset, 
                epoch, 
                device, 
                model, 
                self.args
            )

            print(f"Epoch {epoch+1} | Accuracy: {accuracy}")
            print(f"Epoch {epoch+1} | Average testing loss: {loss}\n")
            print(f"Epoch {epoch+1} | Average AUC: {auc}")
            print(f"Epoch {epoch+1} | Kappa: {kappa}\n")

            test_losses.append(loss)
            accuracies.append(accuracy)
            aucs.append(auc)
            kappas.append(kappa)


            # Quit early if satisfy certain situations
            if accuracy >= self.args.train_until_acc/100:
                print(
                    f"Accuracy reached {self.args.train_until_acc/100} in epoch {epoch+1}, stopping...")
                break
            if self.args.stop_if_improvment_lt:
                if epoch > 0:
                    if accuracies[-2]+self.args.stop_if_improvment_lt/100 >= accuracies[-1]:
                        break

        end = time.time()
        print(f"\nTime used: {time.strftime('%H:%M:%S', time.gmtime(end-start))}")

        # Print final results
        print("Losses on training data:")
        print(f"\t{train_losses}\n")
        print("Losses on testing data:")
        print(f"\t{test_losses}\n")
        print("Accuracies on testing data:")
        print(f"\t{accuracies}\n")
        print("Average AUCs on testing data:")
        print(f"\t{aucs}\n")
        print("Kappas on testing data:")
        print(f"\t{kappas}\n")

        print(f"Final accuracy: {accuracies[-1]}")
        print(f"Final loss: {test_losses[-1]}\n")

        # Write results to file
        if self.args.save_results:
            utils.save_results_to_file(
                self.args,  
                [],
                train_losses, 
                test_losses, 
                accuracies,
                aucs,
                kappas
            )
 
  
if __name__ == "__main__":
    args = Parser().parse()
    if args.learning == "f":
        FederatedLearning(args).run()
    else:
        CentralizedLearning(args).run()
