Random Seed:  8594
totoal data :  1763
prepare train data
len(train_dataset) 1058
len(train_dataloader) 106
totoal data :  1763
prepare validation data
len(validation_dataset) 352
totoal data :  1763
prepare test data
len(test_dataset) 353
module GGNN(
  (in_0): Linear(in_features=40, out_features=40, bias=True)
  (out_0): Linear(in_features=40, out_features=40, bias=True)
)
prefix in_
module GGNN(
  (in_0): Linear(in_features=40, out_features=40, bias=True)
  (out_0): Linear(in_features=40, out_features=40, bias=True)
)
prefix out_
opt Namespace(annotation_dim=1, batchSize=10, cuda=False, dataroot='/home/yiwu/ggnn/wy/ggnn.pytorch/wy_data/all_txt_i/', lr=0.01, manualSeed=8594, n_edge_types=1, n_node=40, n_steps=40, niter=7, question_id=0, state_dim=40, task_id=4, verbal=False, workers=2)
GGNN(
  (in_0): Linear(in_features=40, out_features=40, bias=True)
  (out_0): Linear(in_features=40, out_features=40, bias=True)
  (propogator): Propogator(
    (reset_gate): Sequential(
      (0): Linear(in_features=120, out_features=40, bias=True)
      (1): Sigmoid()
    )
    (update_gate): Sequential(
      (0): Linear(in_features=120, out_features=40, bias=True)
      (1): Sigmoid()
    )
    (tansform): Sequential(
      (0): Linear(in_features=120, out_features=40, bias=True)
      (1): Tanh()
    )
  )
  (out): Sequential(
    (0): Linear(in_features=41, out_features=40, bias=True)
    (1): Tanh()
    (2): Linear(in_features=40, out_features=40, bias=True)
  )
)
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
model two self.n_steps 40
