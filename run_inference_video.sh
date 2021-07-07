echo "DATA_DIR=$DATA_DIR"
echo "TRAINING_DIR=$TRAINING_DIR"
python inference_video.py --labelmap_path label_map.pbtxt --model_path $TRAINING_DIR/exported_model/saved_model/ --tf_record_path $DATA_DIR/processed/test/segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord --config_path $TRAINING_DIR/reference/pipeline_new.config --output_path animation.mp4
