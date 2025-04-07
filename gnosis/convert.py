import os
import zipfile
import argparse
from pathlib import Path
import subprocess
from transformers import AutoTokenizer


def convert_model(model_id, model_type, output_dir, quantize=True):
    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    script = "../ai_edge_torch/generative/examples/llama/convert_to_tflite.py" if model_type == "llama" else "../ai_edge_torch/generative/examples/gemma3/convert_gemma3_to_tflite.py"

    cmd = [
        "python", script,
        "--model_size=1b",
        f"--checkpoint_path={model_id}",
        f"--output_path={output_dir}",
        f"--output_name_prefix=model",
        "--prefill_seq_lens=128",
        "--kv_cache_max_len=128"
    ]
    if not quantize:
        cmd.append("--no_quant")

    print(f"[Run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    tflite_path = os.path.join(output_dir, "model.tflite")
    return tokenizer, tflite_path


def create_task_file(tflite_path, tokenizer_dir, task_path):
    print("[Task] Packaging .task file...")
    with zipfile.ZipFile(task_path, mode="w") as z:
        z.write(tflite_path, arcname="model.tflite")
        for fname in ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
            fpath = Path(tokenizer_dir) / fname
            if fpath.exists():
                z.write(fpath, arcname=f"assets/{fname}")
    print(f"[Task] Saved: {task_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--model_type", required=True, choices=["llama", "gemma"], help="Model type: llama or gemma")
    parser.add_argument("--output_dir", default="export", help="Output directory")
    parser.add_argument("--no_quant", action="store_true", help="Disable quantization")
    args = parser.parse_args()

    tokenizer, tflite_path = convert_model(args.model_id, args.model_type, args.output_dir, not args.no_quant)
    create_task_file(tflite_path, args.output_dir, os.path.join(args.output_dir, "model.task"))


if __name__ == "__main__":
    main()
