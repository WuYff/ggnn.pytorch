import torch
from torch.autograd import Variable

def test(dataloader, net, criterion, optimizer, opt):
    test_loss = 0
    correct = 0
    net.eval()
    for i, (adj_matrix, annotation, target) in enumerate(dataloader, 0):
        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()

        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        target = Variable(target)

        output = net(init_input, annotation, adj_matrix)
        print("test output",output)
        print("test output.shape",output.shape)

        # test_loss += criterion(output, target).data[0]
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        print("test pred.shape",pred.shape)
        print("test pred",output.data.max(1, keepdim=True))
        print("test pred",pred)
  
        correct = 0
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum() #### this has bug

    test_loss /= len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
