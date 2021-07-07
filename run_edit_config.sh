echo "DATA_DIR=$DATA_DIR"
echo "TRAINING_DIR=$TRAINING_DIR"
python edit_config.py --train_dir $DATA_DIR/processed/train/ --eval_dir $DATA_DIR/processed/val/ --batch_size 64 --checkpoint $TRAINING_DIR/pretrained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint0/ckpt-0 --label_map label_map.pbtxt
cp pipeline_new.config $TRAINING_DIR/reference/
