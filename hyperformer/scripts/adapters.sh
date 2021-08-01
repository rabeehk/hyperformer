# This script trains adapters method.

# We train adapters for each individual task of GLUE (by setting the `tasks` and `eval_tasks` 
# to each individual GLUE tasks from `GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"]`.
# Wew tried with reduction factor of `32 and 16` for each of them and report the test results of the model
# obtaining the best results on average across all tasks on the validation sets.
python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/adapters.json 
