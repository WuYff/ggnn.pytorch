import os

path = "/home/yiwu/ggnn/wy/ggnn.pytorch/wy_data/all_txt_i/"
path_list = os.listdir(path)
key = "graph"
a = {0: 0}
for i in range(0, 28):
    a[i * 10] = 0
count_graph = 0
count_ini = 0
for filename in path_list:
    if key in filename:
        file = path + filename[:-10]
        count_graph += 1
        if os.path.exists(file + "_graph.txt"):
            with open(file + "_graph.txt", 'r') as f:
                lines = f.readlines()  # 读取所有行
                last_line = lines[-1]  # 取最后一行
                n = last_line.split(" ")
                node = int(n[0])
                if len(lines) == 3 and node == 3:
                    count_ini += 1
                    print("@init ", filename)
                    print(lines)
                    continue
                for i in range(0, 28):
                    if node <= i * 10:
                        a[i * 10] += 1

# print(a)
for i in range(0, 28):
    a[i * 10] = 100 * a[i * 10] / (count_graph - count_ini)
    a[i * 10] = format(a[i * 10], '.2f')

print(a)
print("ini", count_ini)
print("total grahp without init ", (count_graph - count_ini))
