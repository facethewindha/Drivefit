$env:CUDA_VISIBLE_DEVICES="0"
python train.py --model DiT-XL/2 `
    --data-path ./datasets/Ithaca365/Ithaca365-scenario `
    --boxes-path ./datasets/box_info `
    --epochs 3000 --global-batch-size 4 --lr 1e-5 --log-every 50 --ckpt-every 100 `
    --resume-checkpoint ./results/ithaca365/001-DiT-XL-2/checkpoints/0010400.pt `
    --vae-checkpoint ./pretrained_models/sd-vae-ft-ema `
    --embed-checkpoint ./pretrained_models/clip_similarity_embed.pt `
    --dataset_name ithaca365 --training_sample_steps 500 --scenario_num 5 --rank 2 --modulation `
    --cond_mlp_modulation --rope --finetune_depth 28 --mask_rl 2 --noise_schedule progress