# source activate
# conda activate PDBB

export CUDA_VISIBLE_DEVICES=$5
DPATH="PATH"
 
LR=$2
MODEL=$3
EPOCH=$4
DATASET=$1
 

python Launch_FL.py --root $DPATH \
--exp_name cross_cls --model_name $MODEL --dataset $DATASET \
--num_cls_per_client 1 --num_clusters $i  --num_epoch $EPOCH --avail_percent 1 \
--output-dir PATH/remote/PDBB/outputs --lr $LR  --zsl gzsl

    

