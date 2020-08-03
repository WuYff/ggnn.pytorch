import torch
import torch.nn as nn

import torch.nn.functional as F

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        print("module",module)
        self.prefix = prefix
        print("prefix",prefix)

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types] #[10,8,16]
        A_out = A[:, :, self.n_node*self.n_edge_types:]
        # print("A_in.shape ",A_in.shape)
        # print("A_out.shape ",A_out.shape)

        a_in = torch.bmm(A_in, state_in)  #[10,8,5] = [10,8,16],[10,16,5] 两类边的信息和在了一起
        a_out = torch.bmm(A_out, state_out) # 为什么要相乘呢，是对应着什么公式吗？
        a = torch.cat((a_in, a_out, state_cur), 2) # ([10, 8, 15]
        # print("a_in.shape ",a_in.shape)
        # print("a_out.shape ", a_out.shape)
        # print("a.shape ",a.shape)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        # print("r.shape ",r.shape)
        # print("z.shape ",z.shape)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        # print("joined_input.shape ",joined_input.shape)
        h_hat = self.tansform(joined_input)
        # print("h_hat.shape ",h_hat.shape)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            #nn.Linear(self.state_dim, 1) !!!!!!!!
            nn.Linear(self.state_dim, self.state_dim)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        print("model two self.n_steps",self.n_steps)
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            # print("self.n_edge_types",self.n_edge_types)
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            # print(">in_states ",len(in_states))
            # print(">out_states ",len(out_states))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            # print("]in_states ",in_states.shape)
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            # print("+in_states ",in_states.shape)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            # print("]out_states ",out_states.shape)
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            # print("+out_states ",out_states.shape)

            prop_state = self.propogator(in_states, out_states, prop_state, A)
            # print("+prop_state ",prop_state.shape)

        # print(">prop_state ",prop_state.shape)
        # print(">annotation ",annotation.shape)
        # print("join_state = torch.cat((prop_state, annotation), 2)")
        join_state = torch.cat((prop_state, annotation), 2)
        # print(">>join_state ",join_state.shape)
        output = self.out(join_state)
        # print(">>>output",output.shape) #[batch, v, state_dim]
        # output = output.sum(2) !!!!!!!!!
        # m0 = nn.Softmax(dim=2)
        # output = m0(output)
        output = output.reshape(-1, self.n_node*self.state_dim )

        return output
