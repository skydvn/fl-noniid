from torchvision import datasets, transforms
import math
import random
import torch
import copy
import csv
from datetime import datetime
import os
import numpy as np


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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

        if args.learning == "f":
            if args.iid:
                user_idxs = self.iid_dist(train_dataset, args)
            else:
                user_idxs = self.noniid_dist(train_dataset, args)
        else:
            user_idxs = []

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
        if args.class_per_device:   # Use classes per device
            if args.class_per_device > len(dataset.classes):
                raise OverflowError("Class per device is larger than number of classes")

            if args.equal_dist:
                raise NotImplementedError("Class per device can only be used with unequal distributions")
            else:
                current_classs = 0
                users_classes = [[] for i in range(args.num_devices)]  # Classes dictionary for devices
                classes_devives = [[] for i in range(len(dataset.classes))]  # Devices in each class

                # Distribute class numbers to devices
                for i in range(args.num_devices):
                    next_current_class = (current_classs+args.class_per_device)%len(dataset.classes)
                    if next_current_class > current_classs:
                        users_classes[i] = np.arange(current_classs, next_current_class)
                    else:
                        users_classes[i] = np.append(
                            np.arange(current_classs, len(dataset.classes)),
                            np.arange(0, next_current_class)
                        )
                    
                    for j in users_classes[i]:
                        classes_devives[j].append(i)
                    
                    current_classs = next_current_class

                # Combine indexes and labels for sorting
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
                idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

                users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices

                current_idx = 0
                for i in range(len(dataset.classes)):
                    if not len(classes_devives[i]):
                        continue

                    send_to_device = 0
                    for j in range(current_idx, len(idxs_labels[0])):
                        if idxs_labels[1, j] != i:
                            current_idx = j
                            break

                        users_idxs[classes_devives[i][send_to_device]].append(idxs_labels[0, j])
                        send_to_device = (send_to_device+1)%len(classes_devives[i])

                """
                # Validate results
                tmp_list = []
                for i in range(args.num_devices):
                    tmp_list = list(set(tmp_list) | set(users_idxs[i])) 
                print(len(tmp_list))

                sum = 0
                for i in range(args.num_devices):
                    sum += len(users_idxs[i])
                print(sum)
                """

                return users_idxs

        else:   # Use non-IIDness
            if args.equal_dist:
                data_per_device = math.floor(len(dataset)/args.num_devices)
                users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices

                # Combine indexes and labels for sorting
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
                idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
                idxs = idxs_labels[0, :].tolist()

                niid_data_per_device = int(data_per_device*args.noniidness/100)

                # Distribute non-IID data
                for i in range(args.num_devices):
                    users_idxs[i] = idxs[i*niid_data_per_device:(i+1)*niid_data_per_device]

                # Still have some data
                if args.num_devices*niid_data_per_device < len(dataset):
                    # Filter distributed data
                    idxs = idxs[args.num_devices*niid_data_per_device:]
                    # Randomize data after sorting
                    random.shuffle(idxs)

                    remaining_data_per_device = data_per_device-niid_data_per_device

                    # Distribute IID data
                    for i in range(args.num_devices):
                        users_idxs[i].extend(idxs[i*remaining_data_per_device:(i+1)*remaining_data_per_device])
                
                """
                # Validate results
                for i in users_idxs[0]:
                    print(idxs_labels[1, np.where(idxs_labels[0] == i)])

                sum = 0
                for i in range(args.num_devices):
                    sum += len(users_idxs[i])
                print(sum)
                """
                
                return users_idxs
            else:
                # Max data per device
                max = math.floor(len(dataset)/args.num_devices)
                # Each device get [0.5*max, max) amount of data
                data_per_device = [int(random.uniform(max/2, max)) for i in range(args.num_devices)]

                users_idxs = [[] for i in range(args.num_devices)]  # Index dictionary for devices

                # Combine indexes and labels for sorting
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
                idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
                idxs = idxs_labels[0, :].tolist()

                niid_data_per_device = [int(data_per_device[i]*args.noniidness/100) for i in range(args.num_devices)]

                current_idx = 0
                # Distribute non-IID data
                for i in range(args.num_devices):
                    users_idxs[i] = idxs[current_idx:current_idx+niid_data_per_device[i]]
                    current_idx += niid_data_per_device[i]

                # Filter distributed data
                idxs = idxs[current_idx:]
                # Randomize data after sorting
                random.shuffle(idxs)

                remaining_data_per_device = [data_per_device[i]-niid_data_per_device[i] for i in range(args.num_devices)]

                current_idx = 0
                # Distribute IID data
                for i in range(args.num_devices):
                    users_idxs[i].extend(idxs[current_idx:current_idx+remaining_data_per_device[i]])
                    current_idx += remaining_data_per_device[i]

                """
                # Validate results
                tmp_list = []
                for i in range(args.num_devices):
                    tmp_list = list(set(tmp_list) | set(users_idxs[i])) 
                print(len(tmp_list))

                sum = 0
                for i in range(args.num_devices):
                    sum += len(users_idxs[i])
                print(sum)
                """

                return users_idxs

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

        key = list(w2.keys())

        for key in list(w2.keys()):
            w2[key] = w2[key].reshape(-1).tolist()    # Reshape to 1d tensor and transform to list

        # List for differences: for all devices, get the average of abs((val(device)-val(average))/val(average))
        diff_list = []  
        print("\n\tWeight difference:")

        for key in list(w2.keys()):
            tmp2 = []
            for i in range(len(w)):
                tmp = []
                w[i][key] = w[i][key].reshape(-1).tolist()    # Reshape to 1d tensor and transform to list

                for j in range(len(w[i][key])):
                    tmp.append(abs((w[i][key][j]-w2[key][j])/w2[key][j]))   # Abs((val(device)-val(average))/val(average))

                average = sum(tmp)/len(tmp) # Calculate average
                tmp2.append(average) 
                print(f"\t\tWeight difference | Weight {i + 1} | {key} | {average}")

            average = sum(tmp2)/len(tmp2) # Calculate average
            diff_list.append(average) 

        return sum(diff_list)/len(diff_list)

    def save_results_to_file(self, args, avg_weights_diff,
                             global_train_losses, global_test_losses,
                             global_accuracies):
        iid = "iid" if args.iid else "niid"

        # Create folder if it doesn't exist
        if not os.path.exists("results"):
            os.mkdir("results")

        f = open(f"./results/{datetime.now()}_{args.dataset}_{args.optim}_{iid}.csv", "w")

        with f:
            if args.learning == "f":
                fnames = ["round", "average weight differences",
                        "train losses", "test losses", "test accuracies"]
            else:
                fnames = ["round", "train losses", 
                        "test losses", "test accuracies"]
            writer = csv.DictWriter(f, fieldnames=fnames)

            writer.writeheader()

            for i in range(len(avg_weights_diff)):
                if args.learning == "f":
                    writer.writerow({
                        "round": i+1,
                        "average weight differences": avg_weights_diff[i],
                        "train losses": global_train_losses[i],
                        "test losses": global_test_losses[i],
                        "test accuracies": global_accuracies[i]
                    })
                else:
                    writer.writerow({
                        "round": i+1,
                        "train losses": global_train_losses[i],
                        "test losses": global_test_losses[i],
                        "test accuracies": global_accuracies[i]
                    })
        print(f"Results stored in results/{datetime.now()}_{args.dataset}_{args.optim}_{iid}.csv")
