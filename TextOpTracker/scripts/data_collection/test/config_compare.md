
To compare configs between pickle snapshot & Task config:

```bash
python config_compare.py \
    --pickle_path /home/user/CodeSpace/HumanoidCtrl/TextOp/TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/params \
    --task_name Tracking-Flat-G1-ProjGravObs-MNMLP-v0 \
    --entry_point rsl_rl_cfg_entry_point \
    --output comparison_report.txt
```