import time
from arg_parser import Parser
from utils import Utils
import torch
import numpy as np
import random
from tqdm import tqdm
from local_trainer import Trainer, Tester
import copy
from models import *


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

        self.utils = Utils()

        # Get dataset and data distribution over devices
        train_dataset, test_dataset, device_idxs = self.utils.get_dataset_dist(self.args)

        # Get number of classes (MNIST: 10, CIFAR100: 100)
        if self.args.dataset == "mnist":
            num_classes = 10
        else:
            num_classes = 100
        
        # Set training model (VGG19/LeNet/ResNet18)
        if self.args.model == "vgg19":
            model = VGG("VGG19", num_classes)
        elif self.args.model == "lenet":
            model = LeNet(num_classes)
        else:
            model = ResNet18(num_classes)

        avg_weights_diff = []
        global_train_losses = []
        global_test_losses = []
        global_accuracies = []

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

            print(f"\tDevices selected: {train_devices}\n")

            # Train on each device and return weights and loss
            for device_num in train_devices:
                weights, loss = Trainer().train(
                    train_dataset, 
                    device_idxs[device_num], 
                    round, 
                    device_num, 
                    device, 
                    copy.deepcopy(model),   # Avoid continuously training same model on different devices
                    args
                )

                local_weights.append(weights)
                local_losses.append(loss)

            avg_weights = self.utils.fed_avg(local_weights) # Federated averaging

            model.load_state_dict(avg_weights)  # Load new weights

            avg_weight_diff = self.utils.cal_avg_weight_diff(local_weights, avg_weights) 
            avg_loss = sum(local_losses)/len(local_losses)

            print(f"\n\tRound {round+1} | Average weight difference: {avg_weight_diff}")
            print(f"\tRound {round+1} | Average training loss: {avg_loss}\n")

            global_train_losses.append(avg_loss)
            avg_weights_diff.append(avg_weight_diff)



            # Test step
            print(f"Round {round+1}  Testing:")
            accuracy, loss = Tester().test(
                test_dataset, 
                round, 
                device, 
                copy.deepcopy(model), 
                args
            )

            print(f"\tRound {round+1} | Average accuracy: {accuracy}")
            print(f"\tRound {round+1} | Average testing loss: {loss}\n")

            global_test_losses.append(loss)
            global_accuracies.append(accuracy)



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

        print(f"Final accuracy: {global_accuracies[-1]}")
        print(f"Final loss: {global_test_losses[-1]}\n")

        # Write results to file
        if self.args.save_results:
            self.utils.save_results_to_file(
                self.args, 
                avg_weights_diff,
                global_train_losses, 
                global_test_losses, 
                global_accuracies
            )

if __name__ == "__main__":
    args = Parser().parse()
    FederatedLearning(args).run()
