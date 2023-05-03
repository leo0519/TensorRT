set -xe
mkdir -p log

for INT8_MODE in IMPLICIT; do
    for SKLN_PREC in INT8; do
        PYSCRIPT=""
        ENV_PARAM=""
        if [ "$INT8_MODE" == "IMPLICIT" ]; then
            PYSCRIPT="builder_varseqlen.py"
            if [ "$SKLN_PREC" == "INT8" ]; then
                export SKLN_INT8=TRUE
            else
                unset SKLN_INT8
            fi
        elif [ "$SKLN_PREC" == "FP16" ]; then
            PYSCRIPT="builder_varseqlen_qdq.py"
        else
            PYSCRIPT="builder_varseqlen_qdq_sklnint8.py"
        fi
        FN=${INT8_MODE,,}-${SKLN_PREC,,}

        # Build TRT engine
        $ENV_PARAM; \
        python3 $PYSCRIPT \
            -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx \
            -o engines/${FN}.engine \
            -b 1 -s 256 --int8 --fp16 --verbose \
            -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 \
            -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt \
            2>&1 | tee log/${FN}-build.log
        
        # Get the precision
        python3 inference_varseqlen.py \
            -e engines/${FN}.engine \
            -s 256 -sq ./squad/dev-v1.1.json \
            -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt \
            -o ./predictions.json
        python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90 2>&1 | tee log/${FN}-accuracy.log

        # Do benchmark
        python3 perf_varseqlen.py -e engines/${FN}.engine -b 1 -s 256 -w 2000 -i 2000 2>&1 | tee log/${FN}-infer.log

    done
done