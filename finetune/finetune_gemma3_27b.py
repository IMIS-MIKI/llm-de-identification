import torch
import dataloader
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# =============================================================================
# CONFIGURATION - Gemma 3 27B BF16
# =============================================================================

# Paths
RAW_DIR = 'raw'
DATA_DIR = 'data'
OUTPUT_DIR = './gemma3-27b-medical-anonymizer'

# Model
MODEL_NAME = "unsloth/gemma-3-27b-it"  # Full BF16 model - no quantization issues

# Training parameters
MAX_SEQ_LENGTH = 8192
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size: 2 * 4 = 8
LEARNING_RATE = 1e-4
EPOCHS = 3

# LoRA parameters
RANK = 64
LORA_ALPHA = 128  # 2x RANK for stronger adapter influence
LORA_DROPOUT = 0  # Optimized for Unsloth

# Export settings
GGUF_QUANTIZATION = "f16"  # Options: q4_k_m, q5_k_m, q8_0, f16


def print_gpu_info():
    """Display GPU information"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        total_memory = round(gpu_stats.total_memory / 1024**3, 1)
        print(f"GPU: {gpu_stats.name}")
        print(f"Memory: {total_memory} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Current memory usage
        allocated = round(torch.cuda.memory_allocated() / 1024**3, 2)
        reserved = round(torch.cuda.memory_reserved() / 1024**3, 2)
        print(f"Memory allocated: {allocated} GB")
        print(f"Memory reserved: {reserved} GB")
    else:
        print("⚠️ No CUDA GPU detected!")


def load_model_and_tokenizer():
    """Load Gemma 3 27B in BF16 (full precision)"""
    print("=" * 60)
    print("Loading Gemma 3 27B model (BF16 - full precision)")
    print("=" * 60)
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,  # Explicit BF16
            load_in_4bit=False,    # NO 4-bit quantization
        )
        
        print("✅ Model loaded successfully in BF16!")
        print_gpu_info()
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


def setup_lora(model):
    """Configure LoRA for parameter-efficient fine-tuning"""
    print("\n" + "=" * 60)
    print("Setting up LoRA configuration")
    print(f"  Rank: {RANK}")
    print(f"  Alpha: {LORA_ALPHA}")
    print(f"  Alpha/Rank ratio: {LORA_ALPHA/RANK}")
    print("=" * 60)
    
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=RANK,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Critical for 27B on 80GB
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n✅ LoRA configured!")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Total parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ LoRA setup failed: {e}")
        raise


def steps_calc(dataset):
    """Calculate steps required for fine-tuning, based on the dataset"""
    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = len(dataset) / effective_batch

    max_steps = int(EPOCHS * steps_per_epoch)
    warmup_steps = int(max_steps / 10)

    return max_steps, warmup_steps


def train(model, tokenizer, dataset):
    """Run the fine-tuning"""
    max_steps, warmup_steps = steps_calc(dataset)

    print("\n" + "=" * 60)
    print("Starting training")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max steps: {max_steps}")
    print(f"  Dataset size: {len(dataset)}")
    print("=" * 60)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
            report_to="none",
            save_steps=500,
            save_total_limit=2,
            dataloader_pin_memory=False,
        ),
    )
    
    print("\n🚀 Training started...")
    trainer_stats = trainer.train()
    
    print("\n✅ Training completed!")
    print(f"  Total steps: {trainer_stats.global_step}")
    print(f"  Training loss: {trainer_stats.training_loss:.4f}")
    
    return trainer_stats


def save_adapter_only(model, tokenizer, path):
    """Save only the LoRA adapter (small, fast)"""
    adapter_path = f"{path}_adapter"
    print(f"\n📁 Saving LoRA adapter to: {adapter_path}")
    
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    print(f"✅ Adapter saved!")
    return adapter_path


def save_merged_model(model, tokenizer, path):
    """Merge LoRA into base model and save as HuggingFace format"""
    merged_path = f"{path}_merged"
    print(f"\n📁 Merging and saving full model to: {merged_path}")
    
    # Save merged model in 16-bit
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",  # Explicit 16-bit for clean conversion
    )
    
    print(f"✅ Merged model saved!")
    return merged_path


def export_to_gguf(model, tokenizer, path, quantization="q4_k_m"):
    """Export to GGUF format for Ollama/llama.cpp"""
    gguf_path = f"{path}_gguf"
    print(f"\n📁 Exporting to GGUF ({quantization}) to: {gguf_path}")
    
    model.save_pretrained_gguf(
        gguf_path,
        tokenizer,
        quantization_method=quantization,
    )
    
    print(f"✅ GGUF export completed!")
    print(f"\n📋 To use with Ollama:")
    print(f"   1. Create a Modelfile:")
    print(f"      FROM {gguf_path}/unsloth.{quantization.upper()}.gguf")
    print(f"   2. Create the model:")
    print(f"      ollama create my-model -f Modelfile")
    print(f"   3. Run:")
    print(f"      ollama run my-model")
    
    return gguf_path


def export_for_awq(model, tokenizer, path):
    """
    Prepare model for AWQ quantization with llm-compressor.
    
    Note: Unsloth doesn't have native AWQ export.
    This saves the merged 16-bit model, which can then be quantized 
    with llm-compressor (vLLM project) using awq_quantize.py
    """
    awq_path = f"{path}_for_awq"
    print(f"\n📁 Preparing model for AWQ quantization: {awq_path}")
    
    # Save as merged 16-bit (llm-compressor will quantize this)
    model.save_pretrained_merged(
        awq_path,
        tokenizer,
        save_method="merged_16bit",
    )
    
    print(f"✅ Model prepared for AWQ!")
    print(f"\n📋 Next step: Run awq_quantize.py")
    print(f"   python awq_quantize.py --model_path {awq_path}")
    
    return awq_path


def main():
    """Main fine-tuning pipeline"""
    print("\n" + "=" * 60)
    print("🚀 Gemma 3 27B Fine-tuning Pipeline")
    print("=" * 60 + "\n")
    
    # Show GPU info
    print_gpu_info()
    
    # Step 1: Load model
    print("\n[1/6] Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 2: Setup LoRA
    print("\n[2/6] Setting up LoRA...")
    model = setup_lora(model)
    
    # Step 3: Load and prepare data
    print("\n[3/6] Loading and preparing data...")
    data_dict = dataloader.load()
    formatted_data = dataloader.convert_to_gemma3_format(data_dict)

    if not formatted_data:
        print("❌ No data to train on!")
        return
    
    dataset = dataloader.create_gemma_dataset(formatted_data, tokenizer)
    if dataset is None:
        print("❌ No data to train on!")
        return

    # Step 4: Train
    print("\n[4/6] Training...")
    trainer_stats = train(model, tokenizer, dataset)
    
    # Step 5: Save adapter (quick backup)
    print("\n[5/6] Saving adapter...")
    save_adapter_only(model, tokenizer, OUTPUT_DIR)
    
    # Step 6: Export
    print("\n[6/6] Exporting models...")
    
    # GGUF for Ollama
    try:
        export_to_gguf(model, tokenizer, OUTPUT_DIR, GGUF_QUANTIZATION)
    except Exception as e:
        print(f"⚠️ GGUF export failed: {e}")
        print("You can try manually later with the saved adapter.")
    
    # Prepare for AWQ (vLLM)
    try:
        export_for_awq(model, tokenizer, OUTPUT_DIR)
    except Exception as e:
        print(f"⚠️ AWQ preparation failed: {e}")
        print("You can try manually later with the saved adapter.")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Pipeline completed!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  📁 LoRA Adapter:    {OUTPUT_DIR}_adapter/")
    print(f"  📁 GGUF (Ollama):   {OUTPUT_DIR}_gguf/")
    print(f"  📁 For AWQ (vLLM):  {OUTPUT_DIR}_for_awq/")


if __name__ == '__main__':
    main()
