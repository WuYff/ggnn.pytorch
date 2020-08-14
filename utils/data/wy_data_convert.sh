#! /bin/bash
gzip /home/yiwu/ggnn/wy/ggnn.pytorch/utils/data/valid.jsonl
gzip /home/yiwu/ggnn/wy/ggnn.pytorch/utils/data/test.jsonl
gzip /home/yiwu/ggnn/wy/ggnn.pytorch/utils/data/train.jsonl
mv /home/yiwu/ggnn/wy/ggnn.pytorch/utils/data/train.jsonl.gz /home/yiwu/ggnn/tf-gnn-samples/data/rdf
mv /home/yiwu/ggnn/wy/ggnn.pytorch/utils/data/test.jsonl.gz /home/yiwu/ggnn/tf-gnn-samples/data/rdf
mv /home/yiwu/ggnn/wy/ggnn.pytorch/utils/data/valid.jsonl.gz /home/yiwu/ggnn/tf-gnn-samples/data/rdf