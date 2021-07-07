echo "DATA_DIR=$DATA_DIR"
rm $DATA_DIR/processed/train/*
rm $DATA_DIR/processed/val/*
rm $DATA_DIR/processed/test/*
python create_splits.py --data_dir $DATA_DIR/processed
