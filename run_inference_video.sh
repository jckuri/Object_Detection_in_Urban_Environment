echo "DATA_DIR=$DATA_DIR"
echo "TRAINING_DIR=$TRAINING_DIR"
python inference_video.py --labelmap_path label_map.pbtxt --model_path "$TRAINING_DIR/exported_model/saved_model/" --tf_record_path "$DATA_DIR/processed/test/$1" --config_path "$TRAINING_DIR/reference/pipeline_new.config" --output_path "videos/$1.mp4"
