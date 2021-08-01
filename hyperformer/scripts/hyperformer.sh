# This script trains hyperformer model.

# We experimented with `reduction_factor` of 32, 16 and report the test results of the model obtaining 
# the best validation result on the validation set.
python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/Hyperformer.json 
