# Example script to quantize Llama 3.2 1B Instruct to approx. 2 bits

# Fill these in with your own paths

CKPT="./tmp/models"  # savedir of quantized model weights
HF="./tmp/models"  # savedir of hfized model weights (compatible with transformers library)
LOG="./tmp/logs"  # logdir, I disabled logging
HESS="./tmp/hessians" # dir where you cloned the hessian weights to
APPEND="int8_finetune"  # model name, will be saved as i.e. '/tmp/qtip/models/31_8b_2bit'

# do this if the above dirs don't already exist
#mkdir $CKPT
#mkdir $LOG
#mkdir $HF

# this might take some time...
# main quantization script
# python -m quantize_llama.quantize_finetune_llama \
#        --save_path $CKPT/$APPEND \
#        --codebook bitshift \
#        --base_model meta-llama/Llama-3.2-1B-Instruct \
#        --in_hess_path $HESS \
#        --scale_override 0.9 \
#        --ft_epochs 5 \
#        --td_x 16 \
#        --td_y 16 \
#        --L 16 \
#        --K 2 \
#        --V 1 \
#        --decode_mode 1mad \
#        --tlut_bits 9 \
#        >> $LOG/$APPEND 2>&1

# convert the quantized model to a hf model
# python -m quantize_llama.hfize_llama --quantized_path $CKPT/$APPEND --hf_output_path $HF/$APPEND >> $LOG/$APPEND 2>&1 

# do end to end finetuning
python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-3.2-1B-Instruct --hf_path $HF/$APPEND --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 2048 --ft_train_lut --hf_output_path $HF/$APPEND >> $LOG/$APPEND 2>&1

# evaluate perplexity and zeroshot results
python -m eval.eval_ppl  --hf_path $HF/$APPEND >> $LOG/$APPEND 2>&1
python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/$APPEND >> $LOG/$APPEND 2>&1
