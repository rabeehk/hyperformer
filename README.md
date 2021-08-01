# Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks
This repo contains the PyTorch implementation of the ACL, 2021 paper
[Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks](https://aclanthology.org/2021.acl-long.47.pdf).

# Installation
```
python setup.py install 
```

## How to run the models
We provide example scripts for each model in `hyperformer/scripts/` folder with 
their config files in `hyperformer/configs`. To run the models, please
do `cd hyperformer` and:

 - To run hyperformer++ model (This model generates the task-specific adapters
   using a shared hypernetwork, which is shared across the tasks and layers of a 
   transformer.):
   ```
   bash scripts/hyperformer++.sh
   ``` 
 - To run hyperformer model (This model generates the task-specific adapters using
   a shared hypernetwork, which is shared across the tasks, but this is specific
   to each layer of a transformer. This model is less efficient compared to 
   hyperformer++.):
   ```
   bash scripts/hyperformer.sh
   ``` 
 - To run adapter\dagger model (This model share the layer normalization
   between adapters across the tasks, and train adapters in a multi-task
   setting.):
   ```
   bash scripts/adapters_dagger.sh   
   ``` 
 - To run adapter model (This model trains a single-adapter per task and
   trains the adapters in a single-task learning.):
   ```
   bash scripts/adapters.sh 
   ``` 
 - To run T5 finetuning model in a multi-task learning setup:
   ```
   bash scripts/finetune.sh
   ```

 - To run T5 finetuning model in a single-task learning setup:
   ```
   bash scripts/finetune_single_task.sh
   ```

We run all the models on 4 GPUs, while this is not necessary and one can 
run the models on 1 GPU. In case running on one GPU, in all the scripts, 
please remove the `-m torch.distributed.launch --nproc_per_node=4` part.


## Bibliography
If you find this repo useful, please cite our paper.

```
@inproceedings{karimi2021parameterefficient,
  title={Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks},
  author={Karimi Mahabadi, Rabeeh and Ruder, Sebastian and Dehghani, Mostafa and Henderson, James},
  booktitle={Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```

## Final words
Hope this repo is useful for your research. For any questions, please create an issue or
email rabeeh.k68@gmail.com, and I will get back to you as soon as possible.

