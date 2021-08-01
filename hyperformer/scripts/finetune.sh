# This script trains multi-task T5 on the GLUE benchmark.
python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/finetune.json 
