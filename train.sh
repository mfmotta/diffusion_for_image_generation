export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./mm/input_images"
export PRIOR_PRESERV_DIR="./mm/generated_samples"

export OUTPUT_DIR="./mm_model"  #The output directory where the model predictions and checkpoints will be written.",
export CKPT_DIR="./checkpoint-500"

#y | git config --global credential.helper store
#huggingface-cli login
#TODO save configs in docker run, use alias for huggingface login token

#see https://huggingface.co/docs/diffusers/training/dreambooth

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of mfm" \
  --class_prompt="a photo of woman" \  ###check if best
  --class_data_dir:$PRIOR_PRESERV_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=900 \
  --num_train_epochs=1 \
  --gradient_checkpointing \
  --checkpointing_steps=300 \
  --resume_from_checkpoint=$CKPT_DIR \
  #--push_to_hub \
  #--train_text_encoder  # can't train two models with gradient accumulation see in code: TODO (patil-suraj):
  # can't train 2 models with deepspeed, possible workaround: https://github.com/huggingface/accelerate/issues/253#issuecomment-1253231210