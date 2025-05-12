15 episodes
4 segments
4 skills
256 batch size 
Hidden size 16

lats shape:  (15, 16) --> (eps/batch, hidden_size)
sample_b shape:  (15, 32) --> (eps/batch, max_episode_length)

all lats shape: (4, 15, 16) --> (segments, episodes, hidden_size)

all z shape (4, 15, 16)
all sample bs shape (4, 15) -> lists of length 15. Each list is where episode_n starts for that segment
Collected 60 segments of dim 16
Fitted KMeans with 4 clusters on 60 segments.


```python

actions_np = batch.action.detach().cpu().numpy()
test_np = batch.test.detach().cpu().numpy()
print("batch actions shape", actions_np.shape)
print("batch test shape", test_np.shape)

```


Traceback (most recent call last):
  File "main.py", line 110, in <module>
    app.run(main)
  File "/home/nightless/anaconda3/envs/ompn/lib/python3.6/site-packages/absl/app.py", line 300, in run
    _run_main(main, args)
  File "/home/nightless/anaconda3/envs/ompn/lib/python3.6/site-packages/absl/app.py", line 251, in _run_main
    sys.exit(main(argv))
  File "main.py", line 100, in main
    train_compile.main(training_folder=trainig_folder)
  File "/home/nightless/Desktop/ompn_new/compile/train_compile.py", line 356, in main
    use_id = ((acts == 4).nonzero()).view(-1).cpu()
AttributeError: 'tuple' object has no attribute 'view'


Type of acts during train torch.int64
Type of use_id during train torch.int64