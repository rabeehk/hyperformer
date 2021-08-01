# This scripts trains T5 in a single-task setting.

# We train the model on each single task from the GLUE benchmark by setting the `tasks` and `eval_tasks` 
# to one of GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"], and report the 
# average obtained test scores.
python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/finetune_single_task.json 
