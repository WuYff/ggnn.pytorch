import os
import numpy as np

# Can we assign -1 ? maybe it is better than 0.
# data structure index starts from zero but node id value should start from one.
# output a whole matrix
def load_graphs_from_file(path: str) -> (list, int):
    data_list = []
    max_node_id = 0
    for i in range(0, 4):
        file = path + "e" + str(i)
        print("file: "+"e" + str(i))
        if os.path.exists(file + "_graph.txt"):
            edge_list = []
            label_list = []
            target_list = []
            # [source node, target node]
            with open(file + "_graph.txt", 'r') as f:
                for line in f:

                    line_tokens = line.split(" ")
                    for i in range(1, len(line_tokens)):
                        digits = [int(line_tokens[0]), 1, 0]
                        if line_tokens[i] == "\n":
                            continue
                        node_id = int(line_tokens[i])
                        digits[2] = node_id
                        if node_id > max_node_id:
                            max_node_id = node_id
                        edge_list.append(digits)

            # [node, rd1, rd2,....]
            with open(file + "_target.txt", 'r') as f:
                for line in f:
                    digits = []
                    line_tokens = line.split(" ")
                    for i in range(0, len(line_tokens)):
                        if line_tokens[i] == "\n":
                            continue
                        digits.append(int(line_tokens[i]))
                    target_list.append(digits)  # [[,,,][,,][,,]]
            # [node,variable]
            with open(file + "node_variable.txt", 'r') as f:
                for line in f:
                    digits = [0, 0]
                    line_tokens = line.split(" ")
                    digits[0] = int(line_tokens[0])
                    digits[1] = int(line_tokens[1])
                    label_list.append(digits)
            data_list.append([edge_list, label_list, target_list])
    return (data_list, max_node_id)


def split_set(data_list):
    n_examples = len(data_list)
    idx = range(n_examples)
    train = idx[:1]
    val = idx[-1:]
    return np.array(data_list)[train], np.array(data_list)[val]


def data_convert(data_list: list, n_annotation_dim: int, n_nodes: int):
    n_tasks = 1
    task_data_list = []
    for i in range(n_tasks):
        task_data_list.append([])
    for item in data_list:
        edge_list = item[0]
        label_list = item[1]
        target_list = item[2]
        task_type = 1
        task_output = create_task_output(target_list, n_nodes)  # 原来是一个int，现在变成了长度为node_n * node_n 的 list
        annotation = np.zeros([n_nodes, n_annotation_dim])
        # annotation[target[1] - 1][0] = 1  # 你需要自己定义 annotation 和  n_annotation_dim
        annotation = create_annotation_output(label_list, annotation)
        task_data_list[task_type - 1].append([edge_list, annotation, task_output])
    return task_data_list

# Notice that the rd_id >= 1. Because zero means the corresponding node does not reach the current node 
# return target[ r0_1,r0_2,.....rn_1, ..., rn_n] with length = V*V
def create_task_output(target_list: list, n_nodes: int) -> np.array:
    a = np.zeros((n_nodes, n_nodes))
    for each_node_rd in target_list:
        for rd_id in each_node_rd[1:]:
            a[each_node_rd[0] - 1][rd_id - 1] = 1

    b = np.zeros(n_nodes*n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            b[i*n_nodes+j] = a[i][j]
    return b

# return annotation matrix [V,  1] (current annotation dim =1)
def create_annotation_output(label_list: list, annotation):
    for each_node_varible in label_list:
        annotation[each_node_varible[0] - 1][0] = each_node_varible[1]
    return annotation

# return adjacency matrix[V,V]
def create_adjacency_matrix(edges, n_nodes, n_edge_types):  # 我感觉应该就是一个点的in边 和 out边都记录了
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx - 1][(e_type - 1) * n_nodes + src_idx - 1] = 1
        a[src_idx - 1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] = 1
    return a


class bAbIDataset():
    """
    Load bAbI tasks for GGNN
    """

    def __init__(self, path, task_id, is_train):
        self.n_edge_types = 1
        self.n_tasks = 1
        all_data, self.n_node = load_graphs_from_file(path)
        all_task_train_data, all_task_val_data = split_set(all_data)

        if is_train:
            all_task_train_data = data_convert(all_task_train_data, 1, self.n_node)
            self.data = all_task_train_data[task_id]
        else:
            all_task_val_data = data_convert(all_task_val_data, 1, self.n_node)
            self.data = all_task_val_data[task_id]

    def __getitem__(self, index):
        am = create_adjacency_matrix(self.data[index][0], self.n_node, self.n_edge_types)
        annotation = self.data[index][1]
        target = self.data[index][2]  # my: list , his: int
        return am, annotation, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_dataset = bAbIDataset("", 0, True)
    am, annotation, target = train_dataset.__getitem__(0);
    print("am", am) # [v,v]
    print("annotation", annotation) # [v,1]
    print("target", target) #[v*v]

## 写文档！