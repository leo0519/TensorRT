set -xe

FC_MODE=${1:-"default"}

FLAGS=""
if [ "${FC_MODE}" == "condition1" ]
then
    FLAGS+=" --fc-mode condition1 "
elif [ "${FC_MODE}" == "condition2" ]
then
    FLAGS+=" --fc-mode condition2 "
elif [ "${FC_MODE}" != "default" ]
then
    echo "Unsupport FC mode. Only support [condition1|condition2] or remain empty."
    exit
fi

mkdir -p engines

python3 builder_varseqlen.py \
    -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 \
    -b 1 -s 384 -o engines/megatron_large_seqlen384_int8qat_sparse_${FC_MODE}.engine \
    --fp16 --int8 --strict -il --megatron \
    --pickle models/fine-tuned/bert_pyt_statedict_megatron_sparse_int8qat_v21.03.0/bert_pyt_statedict_megatron_sparse_int8_qat \
    -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -sp \
    ${FLAGS}

python3 inference_varseqlen.py \
    -e engines/megatron_large_seqlen384_int8qat_sparse_${FC_MODE}.engine \
    -s 384 -sq ./squad/dev-v1.1.json \
    -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt \
    -o ./predictions.json

python3 squad/evaluate-v1.1.py squad/dev-v1.1.json ./predictions.json 90