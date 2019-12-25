from torchvision import datasets, transforms
import math
import random
import torch
import copy
import csv
from datetime import datetime
import os


class Utils():
    def get_dataset_dist(self, args):
        if args.dataset == "mnist":
            data_dir = "./data/mnist"

            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            train_dataset = datasets.MNIST(
                root=data_dir,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = datasets.MNIST(
                root=data_dir,
                train=False,
                download=True,
                transform=transform
            )

        elif args.dataset == "cifar100":
            data_dir = "./data/cifar100"

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_dataset = datasets.CIFAR100(
                root=data_dir,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = datasets.CIFAR100(
                root=data_dir,
                train=False,
                download=True,
                transform=transform
            )

        if args.iid:
            user_idxs = self.iid_dist(train_dataset, args)
        else:
            user_idxs = self.noniid_dist(train_dataset, args)

        return train_dataset, test_dataset, user_idxs

    def iid_dist(self, dataset, args):
        data_per_device = math.floor(len(dataset)/args.num_devices)
        idxs = list(range(len(dataset)))    # Put all data index in list
        users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices
        random.shuffle(idxs)
        for i in range(args.num_devices):
            users_idxs[i] = idxs[i*data_per_device:(i+1)*data_per_device]

        return users_idxs

    def noniid_dist(self, dataset, args):
        """
        if args.class_per_device:   # Use classes per device
            args.equal_dist
        else:   # Use non-IIDness
            if args.equal_dist:
                data_per_device = math.floor(len(dataset)/args.num_devices)
                # Sort by labels
            args.noniidness
        """

    def fed_avg(self, weights):
        w = copy.deepcopy(weights[0])   # Weight from first device
        for key in w.keys():
            for i in range(1, len(weights)):    # Other devices
                w[key] += weights[i][key]   # Sum up weights
            w[key] = torch.div(w[key], len(weights))    # Get average weights 
        return w

    def cal_avg_weight_diff(self, weights_list, avg_weights):
        w = copy.deepcopy(weights_list)
        w2 = copy.deepcopy(avg_weights)

        key = list(w2.keys())[0]    # Only take first weight to calculate

        w2[key] = w2[key].reshape(1, -1)[0].tolist()    # Reshape to 1d tensor and transform to list

        # List for differences: for all devices, get the average of abs((val(device)-val(average))/val(average))
        diff_list = []  
        print("\n\tWeight difference:")

        for i in range(len(w)):
            tmp = []
            w[i][key] = w[i][key].reshape(1, -1)[0].tolist()    # Reshape to 1d tensor and transform to list

            for j in range(len(w[i][key])):
                tmp.append(abs((w[i][key][j]-w2[key][j])/w2[key][j]))   # Abs((val(device)-val(average))/val(average))

            average = sum(tmp)/len(tmp) # Calculate average
            diff_list.append(average) 
            print(f"\t\tWeight difference | Weight {i} | {key} | {average}")

        return sum(diff_list)/len(diff_list)

    def save_results_to_file(self, args, avg_weights_diff,
                             global_train_losses, global_test_losses,
                             global_accuracies):
        iid = "iid" if args.iid else "niid"

        # Create folder if it doesn't exist
        if not os.path.exists("results"):
            os.mkdir("results")

        filename = datetime.now() + "_" + args.dataset + "_" + \
            args.optim + "_" + iid + ".csv"
        f = open(f"./results/{filename}", "w")

        with f:
            fnames = ["round", "average weight differences",
                      "train losses", "test losses", "test accuracies"]
            writer = csv.DictWriter(f, fieldnames=fnames)

            writer.writeheader()

            for i in range(len(avg_weights_diff)):
                writer.writerow({
                    "round": i+1,
                    "average weight differences": avg_weights_diff[i],
                    "train losses": global_train_losses[i],
                    "test losses": global_test_losses[i],
                    "test accuracies": global_accuracies[i]
                })

        print(f"Results stored in results/{filename}")
