MODEL_NAME="/datapool/tangzhiyi/data/Qwen3-32B"
TOKENIZER_PATH=/datapool/tangzhiyi/data/Qwen3-32B
SEED=5
NUM_PROMPTS=500
INPUT_LENGTH=8196
PREFIX_LENGTH=6553
OUTPUT_LENGTH=2048
for i in `seq 1 1`
do
    echo "doing"
    python /datapool/tangzhiyi/hetero_ppu/dev/lmdeploy/benchmark/profile_restful_api.py \
    --host 10.201.6.10 \
    --port 8000  \
    --backend vllm \
    --dataset-name random \
    --dataset-path /datapool/tangzhiyi/data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input-len ${INPUT_LENGTH} \
    --random-output-len ${OUTPUT_LENGTH} \
    --model ${MODEL_NAME} \
    --random-range-ratio 0.5 \
    --tokenizer ${TOKENIZER_PATH} \
    --seed ${i} \
    --num-prompts ${NUM_PROMPTS}
done