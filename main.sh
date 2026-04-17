
# #################################
# python3 train_model.py --model unet --batch_size 64 --epochs 300 --lr 0.001 --loss_ablation adaptive_weighted
# python3 train_model.py --model unetpp --batch_size 32 --epochs 150 --lr 0.001 --loss_ablation adaptive_weighted
python3 train_model.py --model attunet --batch_size 16 --epochs 200 --lr 0.003 --loss_ablation dice_ce_focal
# python3 train_model.py --model resunet --batch_size 64 --epochs 300 --lr 0.001 --loss_ablation dice_ce_focal
# python3 train_model.py --model transunet --batch_size 32 --epochs 200 --lr 0.0001 --loss_ablation dice_ce_focal
# python3 train_model.py --model transfuse --batch_size 32 --epochs 200 --lr 0.0001 --loss_ablation dice_ce_focal
# python3 train_model.py --model swinunet --batch_size 16 --epochs 100 --lr 0.001
# python3 train_model.py --model acc_unet --batch_size 2 --epochs 200 --lr 0.001 --loss_ablation dice_ce_focal
# python3 train_model.py --model vmunet --batch_size 4 --epochs 400 --lr 0.001  --loss_ablation adaptive_weighted
# python3 train_model.py --model lightmunet --batch_size 16 --epochs 100 --lr 0.001
# python3 train_model.py --model simpleconvmamba --batch_size 16 --epochs 100 --lr 0.001

# #################################
# python3 train_model.py --model convnext_unet --batch_size 32 --epochs 200 --lr 0.001 --loss_ablation dice_ce_focal
# python3 train_model.py --model mamba_unet --batch_size 8 --epochs 100 --lr 0.001
# python3 train_model.py --model resmamba_unet --batch_size 16 --epochs 100 --lr 0.001
# python3 train_model.py --model convnext_resmamba_add --batch_size 16 --epochs 100 --lr 0.001
# python3 train_model.py --model convnext_resmamba_concat --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal
# python3 train_model.py --model convnext_resmamba_attention --batch_size 8 --epochs 100 --lr 0.001
# python3 train_model.py --model convnext_resmamba_single_udaf --batch_size 8 --epochs 200 --lr 0.001 --loss_ablation dice_ce_focal
# python3 train_model.py --model convnext_resmamba_udaf --batch_size 16 --epochs 100 --lr 0.001

# #################################
# (1.1) Dice-CE
# python3 train_model.py --model convnext_resmamba_udaf_dice_ce --batch_size 8 --epochs 200 --lr 0.001 --loss_ablation dice_ce

# (1.2) Dice-CE + Focal
# python3 train_model.py --model convnext_resmamba_udaf_dice_ce_focal --batch_size 8 --epochs 200 --lr 0.001 --loss_ablation dice_ce_focal

# (1.3) Dice-CE + Boundary Loss
# python3 train_model.py --model convnext_resmamba_udaf_dice_ce_boundary --batch_size 8 --epochs 200 --lr 0.001 --loss_ablation dice_ce_boundary

# (1.4) Dice-CE + Focal Tversky
# python3 train_model.py --model convnext_resmamba_udaf_dice_ce_focal_tversky --batch_size 8 --epochs 200 --lr 0.001 --loss_ablation dice_ce_focal_tversky

# (1.5) Dice-CE + Focal Tversky + Align Loss（lambda消融）
# python3 train_model.py --model convnext_resmamba_udaf --batch_size 8 --epochs 50 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0
# python3 train_model.py --model convnext_resmamba_udaf_align005 --batch_size 8 --epochs 200 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.05
# python3 train_model.py --model convnext_resmamba_udaf_align01 --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1
# python3 train_model.py --model convnext_resmamba_udaf_align02 --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.2

#################################
# python3 train_model.py --model convnext_resmamba_udaf_se --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --post_attn se
# python3 train_model.py --model convnext_resmamba_udaf_eca --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --post_attn eca
# python3 train_model.py --model convnext_resmamba_udaf_cbam --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --post_attn cbam
# python3 train_model.py --model convnext_resmamba_udaf_danet --batch_size 2 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --post_attn danet
# python3 train_model.py --model convnext_resmamba_udaf_coord --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --post_attn coord
# python3 train_model.py --model convnext_resmamba_udaf_epsa --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --post_attn epsa
# python3 train_model.py --model convnext_resmamba_udaf_triplet --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --post_attn triplet
# python3 train_model.py --model convnext_resmamba_udaf_paea --batch_size 2 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --post_attn paea

# #################################
# python3 train_model.py --model convnext_resmamba_udaf_align01 --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --seq T2
# python3 train_model.py --model convnext_resmamba_udaf_align01 --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --seq C2
# python3 train_model.py --model convnext_resmamba_udaf_align01 --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --seq C5
# python3 train_model.py --model convnext_resmamba_udaf_align01 --batch_size 8 --epochs 100 --lr 0.001 --loss_ablation dice_ce_focal_tversky_align --align_lambda 0.1 --seq C2+C5
