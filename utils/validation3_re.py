import torch
from torch.autograd import Variable

def validation(dataloader, net, criterion, optimizer, opt):
    test_loss = 0
    correct = 0
    net.eval()
    each_accurary=0
    total_number=0
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
        # print("Validation max_node_of_one_graph.shape",max_node_of_one_graph.shape)
        # print("Validation max_node_of_one_graph",max_node_of_one_graph)
        
      
        # test_loss += criterion(output, target).data[0]
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        # print("test pred.shape",pred.shape)
        # print("test pred",output.data.max(1, keepdim=True))
        # print("test pred",pred)
        
        
       
        # correct = 0
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum() #### this has bug

        #opt.n_node
        for b in range(len(target)):
            one_correct =0 
            zero_correct =0 
            the_max_node = max_node_of_one_graph[b]
            v_one = 0 
            t_one=0
            # print("@output.shape ",output[b].shape)
            # print("@target.shape ",target[b].shape)
            total_number+=1
            for x in range(the_max_node ): 
                for m in range (the_max_node ):    
                    n = x * opt.n_node + m      
                    if  target[b][n]==1 and output[b][n] > 0 :
                        correct +=1
                        one_correct +=1
                    if ( target[b][n]==0 and output[b][n] <=0 ) :
                        correct +=1
                        zero_correct +=1
                    if  target[b][n]==1:
                        v_one  +=1
            for j in range(len(target[b])):
                if  target[b][j]==1:
                        t_one +=1
            if t_one != v_one:
                print("Somthing Wrong")
            # for n in range(the_max_node,len(target[b])):
            #     if  target[b][n] != 0:
            #         print("Somthing Wrong")    
            # print("Validation v_one",v_one )
            # print("Validation t_one",t_one )
            # print("Validation the_max_node",the_max_node )
            # print("Validation one_correct",one_correct)
            # print("Validation zero_correct",zero_correct)
            # print("the_max_node*the_max_node",the_max_node.item()*the_max_node.item())
            # print("Validation each_correct",(zero_correct+one_correct)/ (the_max_node.item()*the_max_node.item()) )
            each_accurary += (zero_correct+one_correct)/ (the_max_node.item()*the_max_node.item())
    test_loss /= len(dataloader.dataset)
    # print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(test_loss, correct, total_one, 1.0* 100* correct / total_one))
    print('Validation set: Average loss: {:.4f}'.format(test_loss))
    print("Validation (len(dataloader.dataset)", (len(dataloader.dataset)))
    print("Validation (total_number)",total_number)
    print("Validation Average Accuracy :({:.4f}%):".format( 100*each_accurary/len(dataloader.dataset)))