# This scripts trains the Adapter\dagger model.

# we tried with reduction factor of 16, 32 in the config file and reports the test results on the model 
# obtaining the best results on on the validation set.
python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/adapter_dagger.json 
