import torch
from torch.autograd import Variable

def test(dataloader, net, criterion, optimizer, opt):
    test_loss = 0
    correct = 0
    total_one =0 
    net.eval()
    each_accurary=0
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
        print("target.shape",target.shape)

        # test_loss += criterion(output, target).data[0]
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        # print("test pred.shape",pred.shape)
        # print("test pred",output.data.max(1, keepdim=True))
        # print("test pred",pred)
        print("len(target)",len(target))
        print("len(target)",len(target[0]))
        # correct = 0
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum() #### this has bug
        one_correct =0 
        this_one = len(target)*len(target[0])
        total_one += this_one
        for b in range(len(target)):
            for n in range(len(target[0])):
                if ( target[b][n]==1 and output[b][n] >= 0.5 ) or ( target[b][n]==0 and output[b][n] < 0.5 ) :
                    correct +=1
                    one_correct +=1
        each_accurary += 1.0*one_correct/this_one
    test_loss /= len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(test_loss, correct, total_one, 1.0* 100* correct / total_one))
    print(" (len(dataloader.dataset)", (len(dataloader.dataset)))
    print("Average Accuracy :({:.4f}%):".format(each_accurary/len(dataloader.dataset)))