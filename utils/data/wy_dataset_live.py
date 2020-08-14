import os
import numpy as np

# Can we assign -1 ? maybe it is better than 0.
# data structure index starts from zero but node id value should start from one.
# output a whole matrix
def load_graphs_from_file(path: str,how_many:int) -> (list, int):
    data_list = []
    max_node_id = 0
    path_list=os.listdir(path)
    key="graph"
    skip_init = True
    count_ini =0 
    max_def_id = 0
    for filename in path_list:
        if key in filename:
            file = path + "/"+filename[:-10]
            # print("@file: "+filename[:-10])
            if os.path.exists(file + "_graph.txt"):
                edge_list = []
                use_list = []
                def_list = []
                target_list = []
                max_node_of_one_graph = 0
                max_def_of_one_graph =0
                # 
                # [source node, target node]
                with open(file + "_graph.txt", 'r') as f:
                    lines = f.readlines()  # 读取所有行
                    last_line = lines[-1]  # 取最后一行
                    n = last_line.split(" ")
                    node = int(n[0])
                    if skip_init  and ((len(lines) <= 3) or node > how_many ):
                        count_ini += 1
                        continue
                    for line in lines:
                        line_tokens = line.split(" ")
                        for i in range(1, len(line_tokens)):
                            
                            if line_tokens[0] == "":
                                continue
                            digits = [int(line_tokens[0]), 1, 0]
                            if line_tokens[i] == "\n":
                                continue
                            node_id = int(line_tokens[i])
                            digits[2] = node_id
                            if node_id > max_node_id:
                                max_node_id = node_id
                            if node_id > max_node_of_one_graph:
                                max_node_of_one_graph = node_id
                            edge_list.append(digits)
                # print("digits",digits)

                # [node, rd1, rd2,....]
                with open(file + "_target.txt", 'r') as f2:
                    for line in f2:
                        digits = []
                        line_tokens = line.split(" ")
                        for i in range(0, len(line_tokens)):
                            if line_tokens[i] == "\n":
                                # print("right here!")
                                continue
                            digits.append(int(line_tokens[i]))
                        target_list.append(digits)  # [[,,,][,,][,,]]
                # [node,variable]
                with open(file + "_use.txt", 'r') as f3:
                    for line in f3:
                        digits = []
                        line_tokens = line.split(" ")
                        # if len(line_tokens) == 4:
                        #     print("@@line_tokens",line_tokens)
                            
                        # digits[0] = int(line_tokens[0])
                        for j in range(len(line_tokens)-1):
                            digits.append( int(line_tokens[j]))

                        # if len(line_tokens) == 4:
                        #     print("line_tokens",line_tokens)
                        #     digits[2] = int(line_tokens[2])
                        use_list.append(digits)
                with open(file + "_def.txt", 'r') as f4: 
                      for line in f4:
                        digits = [0, 0]
                        line_tokens = line.split(" ")
                        if len(line_tokens) == 4:
                            print("line_tokens",line_tokens)
                            print("Wrong def")
                        digits[0] = int(line_tokens[0])
                        digits[1] = int(line_tokens[1])
                        if digits[0]>max_def_id :
                            max_def_id =digits[0]
                        if digits[1]>max_def_id :
                            max_def_id =digits[1]
                        if digits[0]>max_def_of_one_graph :
                            max_def_of_one_graph =digits[0]
                        if digits[1]>max_def_of_one_graph  :
                            max_def_of_one_graph =digits[1]
                        def_list.append(digits)
                data_list.append([edge_list,def_list, use_list, target_list,max_node_of_one_graph,max_def_of_one_graph])
    print("totoal data : ", len(data_list))
    return (data_list, max_node_id,max_def_id)


def split_set(data_list:list):
    n_examples = len(data_list)
    idx = range(n_examples)
    num = round(n_examples*0.6)
    t=  round(n_examples*0.2)
    train = idx[:num]
    test = idx[num: num+t]
    val =  idx[num+t:]
    f = open("/home/yiwu/ggnn/wy/ggnn.pytorch/utils/data/vali_40.log", 'wt')
    print("################################################", file=f)
    for i in np.array(data_list)[val]:
        print(i, file=f)
    print("################################################", file=f)
    return np.array(data_list)[train], np.array(data_list)[val], np.array(data_list)[test]


def data_convert(data_list: list, n_annotation_dim: int, n_nodes: int):
    n_tasks = 1
    task_data_list = []
    for i in range(n_tasks):
        task_data_list.append([])
    for item in data_list:
        edge_list = item[0]
        def_list=item[1]  # target 和 graph 是全图 
        use_list=item[2]  # def 和 use 都不是全图
        target_list = item[3]
        max_node_of_one_graph = item[4]
        max_def_of_one_graph =  item[5]
        task_type = 1
        task_output = create_task_output(target_list, n_nodes,n_annotation_dim)  # 原来是一个int，现在变成了长度为node_n * node_n 的 list
        annotation = np.zeros([n_nodes, n_annotation_dim])
        # annotation[target[1] - 1][0] = 1  # 你需要自己定义 annotation 和  n_annotation_dim
        annotation = create_annotation_output(def_list, use_list, annotation)
        task_data_list[task_type - 1].append([edge_list, annotation, task_output, max_node_of_one_graph,max_def_of_one_graph])
    return task_data_list

# Notice that the rd_id >= 1. Because zero means the corresponding node does not reach the current node 
# return target[ r0_1,r0_2,.....rn_1, ..., rn_n] with length = V*V
def create_task_output(target_list: list, n_nodes: int,n_def:int ) -> np.array:
    a = np.zeros((n_nodes, n_def))
    # print("n_nodes",n_nodes)
    # print("target_list",target_list)
    for each_node_rd in target_list:
        # print("each_node_rd",each_node_rd)
        for rd_id in each_node_rd[1:]:
            a[each_node_rd[0] - 1][rd_id - 1] = 1
    # print("a>",a)

    b = np.zeros(n_nodes*n_def)
    for i in range(n_nodes):   
        for j in range(n_def):
            b[i*n_def+j] = a[i][j]
    # print("b>",b)
    return b

# return annotation matrix [V,  1] (current annotation dim =1)
def create_annotation_output(def_list: list, use_list: list, annotation):
    for each_node_varible in def_list:
        for i in range(1,len(each_node_varible)):
            annotation[each_node_varible[0] - 1][each_node_varible[i]-1] = -1
    for each_node_varible in use_list:
        for i in range(1,len(each_node_varible)):
            annotation[each_node_varible[0] - 1][each_node_varible[i]-1] = 1
    # 同一个variable可能同时出现，所以要先def kill掉，再 use加进去
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

    def __init__(self, path, task_id, is_train,node_number:int,how_many:int):
        self.n_edge_types = 1
        self.n_tasks = 1
        all_data, self.n_node ,self.n_def = load_graphs_from_file(path,how_many)
        print(" self.n_node ,self.n_def ", self.n_node ,self.n_def )
        all_task_train_data, all_task_val_data, all_task_test_data = split_set(all_data)

        if is_train == "t":
            print("prepare train data")
            all_task_train_data = data_convert(all_task_train_data, self.n_def, self.n_node)
            self.data = all_task_train_data[task_id]
        
        elif is_train == "v":
            print("prepare validation data")
            self.n_node = node_number
            all_task_val_data = data_convert(all_task_val_data, self.n_def, self.n_node)
            self.data = all_task_val_data[task_id]
        else:
            print("prepare test data")
            self.n_node = node_number
            all_task_test_data = data_convert(all_task_test_data, self.n_def, self.n_node)
            self.data = all_task_test_data[task_id]
            
        

    def __getitem__(self, index):
        am = create_adjacency_matrix(self.data[index][0], self.n_node, self.n_edge_types)
        annotation = self.data[index][1]
        target = self.data[index][2]
        max_node_of_one_graph = self.data[index][3]  # my: list , his: int
        max_def_of_one_graph = self.data[index][4] 
        return am, annotation, target, max_node_of_one_graph,max_def_of_one_graph

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_dataset = bAbIDataset("", 0, True)
    am, annotation, target, max_node_of_one_graph= train_dataset.__getitem__(0)
    print("am", am) # [v,v]
    print("annotation", annotation) # [v,1]
    print("target", target) #[v*v]

## 写文档！