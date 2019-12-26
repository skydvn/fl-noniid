import argparse


class Parser():
    def parse(self):
        parser = argparse.ArgumentParser()

        # Training setting
        parser.add_argument("--bs", help="mini batch size",
                            default=64, type=int)
        parser.add_argument("--epoch", help="local epochs per communication round", 
                            default=10, type=int)
        parser.add_argument("--round", help="communication rounds", 
                            default=10, type=int)
        parser.add_argument("--lr", help="learning rate",
                            default=0.01, type=float)
        parser.add_argument("--lr_decay", help="learning rate decay", 
                            default=1, type=float)
        parser.add_argument("--sgd_momentum", help="sgd momentum", 
                            default=0.5, type=float)
        parser.add_argument("--optim", help="optimizer (sgd, adagrad, adam)", 
                            default="sgd", type=str)
        parser.add_argument("--dataset", help="dataset (mnist, cifar100)", 
                            default="mnist", type=str)
        parser.add_argument("--model", help="model (lenet, resnet18, vgg19)", 
                            default="lenet", type=str)
        parser.add_argument("--frac", help="fraction of devices", 
                            default=0.1, type=float)
        parser.add_argument("--num_devices", help="number of devices", 
                            default=100, type=int)

        # IIDness setting
        parser.add_argument("--iid", help="data is iid or not", 
                            default=1, type=int)
        parser.add_argument("--noniidness", help="percentage of non-iid data per device", 
                            default=50, type=float)
        parser.add_argument("--equal_dist", help="equal data distribution across devices", 
                            default=1, type=int)
        parser.add_argument("--class_per_device", help="number of classes per device (non-iid, conflict with noniidness)", 
                            default=0, type=int)

        # Train until options
        parser.add_argument("--train_until_acc", help="train until accuracy reaches x%", 
                            default=100, type=float)
        parser.add_argument("--stop_if_improvment_lt", help="stop if improvement less than x% between two rounds", 
                            default=0, type=float)

        # Other options
        parser.add_argument("--random_seed", help="use random seed or not", 
                            default=0, type=int)
        parser.add_argument("--gpu", help="gpu ID", default=None)
        parser.add_argument("--verbose", help="verbose logging", 
                            default=0, type=int)
        parser.add_argument("--save_results", help="save results to file", 
                            default=1, type=int)

        return parser.parse_args()
