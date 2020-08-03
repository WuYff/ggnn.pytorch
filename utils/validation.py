import torch
from torch.autograd import Variable

def validation(dataloader, net, criterion, optimizer, opt):
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
        
      
        # test_loss += criterion(output, target).data[0]
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        # print("test pred.shape",pred.shape)
        # print("test pred",output.data.max(1, keepdim=True))
        # print("test pred",pred)
        
        
       
        # correct = 0
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum() #### this has bug
        
        this_one = len(target)*len(target[0])
        total_one += this_one
        for b in range(len(target)):
            one_correct =0 
            zero_correct =0 
            print("@output ",output[b])
            print("@target ",target[b])
            for n in range(len(target[b])):              
                if  target[b][n]==1 and output[b][n] >= 0.5 :
                    correct +=1
                    one_correct +=1
                if ( target[b][n]==0 and output[b][n] < 0.5 ) :
                    correct +=1
                    zero_correct +=1
            print("one_correct",one_correct)
            print("one_correct",zero_correct)
            each_accurary += (zero_correct+one_correct)/len(target[b])
    test_loss /= len(dataloader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(test_loss, correct, total_one, 1.0* 100* correct / total_one))
    print("Validation (len(dataloader.dataset)", (len(dataloader.dataset)))
    print("Validation Average Accuracy :({:.4f}%):".format( 100*each_accurary/len(dataloader.dataset)))