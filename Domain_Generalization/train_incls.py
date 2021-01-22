import argparse
import os

import random
import torch
# from IPython.core.debugger import set_trace
from torch import nn
# from torch.nn import functional as F
from data import data_helper
## from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from models.loss import IntraClsInfoMax
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
from models.resencoder import resnet18, resnet50


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--alpha", default=0.1, type=float, help="Minimum scale percent")
    parser.add_argument("--beta", default=0.2, type=float, help="Minimum scale percent")
    parser.add_argument("--gamma", default=0.5, type=float, help="Minimum scale percent")
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets,
                        help="Target")  # ["art_painting", "cartoon", "photo", "sketch"]
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                        default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--foldername", default="", type=str, help="Use nesterov")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(pretrained=True, classes=args.n_classes)
        elif args.network == 'resnet50':
            model = resnet50(pretrained=True, classes=args.n_classes)
        else:
            model = resnet18(pretrained=True, classes=args.n_classes)
        self.model = model.to(device)
        self.D_model = IntraClsInfoMax(alpha=args.alpha, beta=args.beta, gamma=args.gamma).to(device)
        # print(self.model)
        # print(self.D_model)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
            len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler([self.model,self.D_model.global_d, self.D_model.local_d], args.epochs, args.learning_rate, args.train_all,
                                                                 nesterov=args.nesterov)
        self.dis_optimizer,self.dis_scheduler = get_optim_and_scheduler([self.D_model.prior_d], args.epochs, args.learning_rate, args.train_all,
                                                                 nesterov=args.nesterov) #args.learning_ratee*1e-3
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        self.max_test_acc = 0.0
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

    def _do_epoch(self,device='cuda'):

        criterion = nn.CrossEntropyLoss()
        self.model.train()
        self.D_model.train()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(
                self.device), d_idx.to(self.device)

            self.optimizer.zero_grad()

            data_flip = torch.flip(data, (3,)).detach().clone()
            data = torch.cat((data, data_flip))
            class_l = torch.cat((class_l, class_l))

            y, M = self.model(data, feature_flag=True)

            # Classification Loss
            class_logit = self.model.class_classifier(y)
            class_loss = criterion(class_logit, class_l)

            # G loss - DIM Loss - P_loss
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)  # Move feature to front position one by one
            class_prime= torch.cat((class_l[1:], class_l[0].unsqueeze(0)), dim=0)
            class_ll = (class_l, class_prime)

            DIM_loss = self.D_model(y, M, M_prime,class_ll)
            P_loss=self.D_model.prior_loss(y)

            DIM_loss = DIM_loss- P_loss
            # DIM_loss=self.beta*(DIM_loss-P_loss)
            loss = class_loss + DIM_loss
            loss.backward()
            self.optimizer.step()

            self.dis_optimizer.zero_grad()
            P_loss=self.D_model.prior_loss(y.detach())
            P_loss.backward()
            self.dis_optimizer.step()

            # Prediction
            _, cls_pred = class_logit.max(dim=1)



            losses = {'class': class_loss.detach().item(), 'DIM': DIM_loss.detach().item(), 'P_loss': P_loss.detach().item()}
            self.logger.log(it, len(self.source_loader),
                            losses,
                            {"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])
            del loss, class_loss, class_logit,DIM_loss

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)

                class_correct = self.do_test(loader)

                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
                if phase == 'test' and class_acc > self.max_test_acc:
                    torch.save(self.model.state_dict(), os.path.join(self.logger.log_path, 'best_{}.pth'.format(phase)))

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, nouse, class_l), _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            class_logit = self.model(data, feature_flag=False)
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct

    def do_training(self):
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.dis_scheduler.step()
            self.logger.new_epoch([*self.scheduler.get_lr(),*self.dis_scheduler.get_lr()])
            self._do_epoch()        # use self.current_epoch
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


def main():
    args = get_args()
    # args.source = ['art_painting', 'cartoon', 'sketch']
    # args.target = 'photo'
    # args.source = ['art_painting', 'cartoon', 'photo']
    # args.target = 'sketch'
    args.source = ['art_painting', 'photo', 'sketch']
    args.target = 'cartoon'
    # args.source = ['photo', 'cartoon', 'sketch']
    # args.target = 'art_painting'
    # --------------------------------------------
    print("Target domain: {}".format(args.target))
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # --------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
