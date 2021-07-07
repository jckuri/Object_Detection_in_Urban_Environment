echo "TRAINING_DIR=$TRAINING_DIR"
python ../models/research/object_detection/model_main_tf2.py --model_dir=$TRAINING_DIR/reference/ --pipeline_config_path=$TRAINING_DIR/reference/pipeline_new.config --checkpoint_dir=$TRAINING_DIR/reference/
