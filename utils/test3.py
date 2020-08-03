import torch
from torch.autograd import Variable

def test(dataloader, net, criterion, optimizer, opt):
    test_loss = 0
    correct = 0
    total_one =0 
    net.eval()
    each_accurary=0
    for i, (adj_matrix, annotation, target,max_node_of_one_graph) in enumerate(dataloader, 0):
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
            the_max_node = max_node_of_one_graph[b]
            print("@Test output ",output[b])
            print("@Test target ",target[b])
            for n in range(the_max_node):              
                if  target[b][n]==1 and output[b][n] >= 0.5 :
                    correct +=1
                    one_correct +=1
                if ( target[b][n]==0 and output[b][n] < 0.5 ) :
                    correct +=1
                    zero_correct +=1
            print("Test the_max_node",the_max_node)
            print("Test one_correct",one_correct)
            print("Test zero_correct",zero_correct)
            each_accurary += (zero_correct+one_correct)/the_max_node
    test_loss /= len(dataloader.dataset)
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(test_loss, correct, total_one, 1.0* 100* correct / total_one))
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    print("Test (len(dataloader.dataset)", (len(dataloader.dataset)))
    print("Test Average Accuracy :({:.4f}%):".format( 100*each_accurary/len(dataloader.dataset)))