## ArtistAuditor



#### Fine-tune Text-to-image model

We used the fine-tuning scripts provided by [🤗diffusers](https://github.com/huggingface/diffusers/) to fine-tune different text-to-image models.

* SD-V2

  ```sh
  accelerate launch train_text_to_image.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --dataset_name="path/to/dataset_dir" \
    --use_ema \
    --seed=1028 \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --mixed_precision="fp16" \
    --num_train_epochs=100 \
    --checkpointing_steps=500 \
    --learning_rate=5e-6 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="path/to/output_dir" \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention
  ```

* SDXL

  ```sh
  accelerate launch train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --dataset_name="path/to/dataset_dir" \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=1 \
    --num_train_epochs=100 --checkpointing_steps=750 \
    --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --seed=1028 \
    --output_dir="path/to/output_dir" \
    --use_8bit_adam
  ```

* Kandinsky

  ```sh
  accelerate launch --mixed_precision="fp16" train_text_to_image_prior.py \
    --pretrained_prior_model_name_or_path="kandinsky-community/kandinsky-2-2-prior" \
    --dataset_name="path/to/dataset_dir" \
    --resolution=512 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=100 \
    --checkpointing_steps=2000 \
    --learning_rate=1e-4 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --seed=1028 \
    --use_8bit_adam \
    --output_dir="path/to/output_dir"
  ```

  ```sh
  accelerate launch --mixed_precision="fp16" train_text_to_image_decoder.py \
    --train_data_dir="path/to/dataset_dir" \
    --pretrained_decoder_model_name_or_path="kandinsky-community/kandinsky-2-2-decoder" \
    --pretrained_prior_model_name_or_path="kandinsky-community/kandinsky-2-2-prior" \
    --seed=1028 \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --max_grad_norm=1 \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="path/to/output_dir"
  ```

  

#### Inference

* SD-V2

  ```sh
  python inference_sdv2.py \
      --model="path/to/finetuned_model" \
      --prompt_file="path/to/prompt_file" \
      --output_dir="path/to/output_dir" \
      --seed=1
  ```

* SDXL

  ```sh
  python inference_sdxl.py \
      --model="stabilityai/stable-diffusion-xl-base-1.0" \
      --ckpt="path/to/lora_weights" \
      --prompt_file="path/to/prompt_file" \
      --output_dir="path/to/output_dir" \
      --seed=1
  ```

* Kandinsky

  ```sh
  python inference_kandi.py \
      --prior="path/to/finetuned_prior" \
      --decoder="path/to/finetuned_decoder" \
      --prompt_file="path/to/prompt_file" \
      --output_dir="path/to/output_dir" \
      --seed=1
  ```

  

#### Discriminator Construction

```sh
python discriminator_train.py 
  --train_dataset_root_dir='path/to/train_dataset' \
  --validate_dataset_root_dir='path/to/validate_dataset' \
  --public_dataset_ori_dir='path/to/ori_public_dataset' \
  --public_dataset_gen_dir='path/to/gen_public_dataset' \
  --model_save_dir='path/to/model_save_dir' \
  --seed=1 
```

