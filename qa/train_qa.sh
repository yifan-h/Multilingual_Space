#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to train a model on SQuAD v1.1 or the English TyDiQA-GoldP train data.

REPO=$PWD
MODEL=${1:-"$REPO/xlm-roberta-large"}
MODEL_KG=${7:-"$REPO/adapter/xlmr_adapter"}
SRC=${2:-squad}
TGT=${3:-xquad}
GPU=${4:-0}
DATA_DIR=${5:-"$REPO/download/"}
OUT_DIR=${6:-"$REPO/outputs-adapter/"}

BATCH_SIZE=4
GRAD_ACC=4

MAXL=384
LR=8e-6
NUM_EPOCHS=2.0
<< "COMMENT"
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
  MODEL_NAME="mbert-KG"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "/cluster/work/sachan/meizhen/multi/models/huggingface_models/xlm-roberta-base/" ]; then
  MODEL_TYPE="xlm-roberta"
  MODEL_NAME="xlm-KG"
fi
COMMENT
MODEL_TYPE="xlm"
MODEL_NAME="xlm-KG"

# Model path where trained model should be stored
MODEL_PATH=$OUT_DIR/$SRC/${MODEL_NAME}_LR${LR}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $MODEL_PATH
# Train either on the SQuAD or TyDiQa-GoldP English train file
if [ $SRC == 'squad' ]; then
  TASK_DATA_DIR=${DATA_DIR}/squad
  TRAIN_FILE=${TASK_DATA_DIR}/train-v1.1.json
  PREDICT_FILE=${TASK_DATA_DIR}/dev-v1.1.json
else
  TASK_DATA_DIR=${DATA_DIR}/tydiqa
  TRAIN_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-train/tydiqa.en.train.json
  PREDICT_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-dev/tydiqa.goldp.en.dev.json
fi

<< "COMMENT"
# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_squad.py \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${MODEL} \
  --model_name ${MODEL_NAME}\
  --modelkg_dir ${MODEL_KG}\
  --do_train \
  --do_eval \
  --data_dir ${TASK_DATA_DIR} \
  --train_file ${TRAIN_FILE} \
  --predict_file ${PREDICT_FILE} \
  --per_gpu_train_batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --num_train_epochs ${NUM_EPOCHS} \
  --max_seq_length $MAXL \
  --doc_stride 128 \
  --save_steps -1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --warmup_steps 500 \
  --output_dir ${MODEL_PATH} \
  --weight_decay 0.0001 \
  --threads 8 \
  --train_lang en \
  --eval_lang en
COMMENT

# predict
bash scripts/predict_qa.sh $MODEL $MODEL_PATH $TGT $GPU $DATA_DIR
