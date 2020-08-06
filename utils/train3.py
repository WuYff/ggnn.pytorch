import torch
from torch.autograd import Variable

def train(epoch, dataloader, net, criterion, optimizer, opt):
    net.train()
    for i, (adj_matrix, annotation, target,max_node_of_one_graph) in enumerate(dataloader, 0):
        net.zero_grad()
        # print("annotation.shape",annotation.shape)
        # print("annotation[0]",annotation[0])
        # print("opt.n_node",opt.n_node)
        # print("opt.state_dim - opt.annotation_dim",opt.state_dim - opt.annotation_dim)
        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        # print("padding.shape",padding.shape)
        init_input = torch.cat((annotation, padding), 2)
        # print("init_input.shape",init_input.shape)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()

        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        target = Variable(target)
        # print("target size",target.shape)


        output = net(init_input, annotation, adj_matrix)
        # print("output size",output.shape)
        print(">!target",target)
        print(">!output",output)
        
        # print("Train max_node_of_one_graph.shape",max_node_of_one_graph.shape)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        
        # if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
        #     print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.data[0]))
        print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.data))