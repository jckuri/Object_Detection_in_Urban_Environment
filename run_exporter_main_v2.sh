echo "TRAINING_DIR=$TRAINING_DIR"
python ../models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path $TRAINING_DIR/reference/pipeline_new.config --trained_checkpoint_dir $TRAINING_DIR/reference/ --output_directory $TRAINING_DIR/exported_model/
