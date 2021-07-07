echo "TRAINING_DIR=$TRAINING_DIR"
rm -rf $TRAINING_DIR
mkdir -p $TRAINING_DIR
mkdir -p $TRAINING_DIR/pretrained_models
mkdir -p $TRAINING_DIR/reference
mkdir -p $TRAINING_DIR/exported_model
MODEL="ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
MODEL_PATH="$TRAINING_DIR/pretrained_models/$MODEL"
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL -O $MODEL_PATH
tar -xf $MODEL_PATH -C $TRAINING_DIR/pretrained_models/
CKPT_DIR=$TRAINING_DIR/pretrained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
mv $CKPT_DIR/checkpoint $CKPT_DIR/checkpoint0
cp pipeline_new.config $TRAINING_DIR/reference/
