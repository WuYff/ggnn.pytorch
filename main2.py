import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model2 import GGNN
from utils.train2 import train
from utils.test import test
from utils.validation import validation
from utils.data.wy_dataset import bAbIDataset
from utils.data.dataloader import bAbIDataloader

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--state_dim', type=int, default=4, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=1, help='propogation steps number of GGNN')
# parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--niter', type=int, default=7, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()

# todo : shuffle before each epoch, specify the number od n_steps

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = '/home/yiwu/ggnn/wy/ggnn.pytorch/wy_data/all_txt_i/'
test_path= '/home/yiwu/ggnn/wy/ggnn.pytorch/wy_data/all_txt_i_test/'

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

def main(opt):
    train_dataset = bAbIDataset(opt.dataroot, opt.question_id, "t",0)
    print("len(train_dataset)",len(train_dataset))
    # for i, (adj_matrix, annotation, target) in enumerate(train_dataset, 0):
        # print("annotation size",annotation.shape)
        # print("adj_matrix size",adj_matrix.shape)
        # print("target int",target)
        # break
    train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=2)
    print("len(train_dataloader)",len(train_dataloader))
    # for i, (adj_matrix, annotation, target) in enumerate(train_dataloader, 0):
    #     print("@annotation size",annotation.shape)
    #     print("@adj_matrix size",adj_matrix.shape)
    #     print("@target size",target.shape)
    #     break
    

    validation_dataset = bAbIDataset(opt.dataroot, opt.question_id, "v", train_dataset.n_node)
    validation_dataloader = bAbIDataloader(validation_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=2)
    print("len(validation_dataset)",len(validation_dataset))

    test_dataset = bAbIDataset(opt.dataroot, opt.question_id, "est", train_dataset.n_node)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=2)
    print("len(test_dataset)",len(test_dataset))
    opt.annotation_dim = 1  # for bAbI
    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node
    opt.state_dim = opt.n_node
    opt.n_steps = opt.n_node

    net = GGNN(opt)
    net.double()
    

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    print("opt",opt)
    print(net)
    for epoch in range(0, opt.niter):
        train(epoch, train_dataloader, net, criterion, optimizer, opt)
        validation(validation_dataloader, net, criterion, optimizer, opt)
        test(test_dataloader, net, criterion, optimizer, opt)


if __name__ == "__main__":
    main(opt)

