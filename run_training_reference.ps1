
# Run training with parameters matching IMM_CelebA.ipynb reference
python e:/imm_vae/imm-main/train_joint.py --prior imm --dataset CelebA --model_type vae --lr_vae 1e-3 --lr_prior 1e-5 --batch_size 128 --epochs 80 --save_dir ./checkpoints_ref
