from datasets import Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from trl import DPOConfig, DPOTrainer
import wandb
import argparse

# コマンドライン引数のパーサーを設定
parser = argparse.ArgumentParser(description="DPO Training Script")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--deepspeed", type=str, help="DeepSpeed configuration file")
parser.add_argument("--log_type", type=str, default="wandb", help="Logging type")
parser.add_argument("--log_project", type=str, default="DPO", help="Logging project name")
parser.add_argument("--tf32", type=str, default="False", help="Enable TF32 precision")
parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Per device batch size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=64, help="Gradient accumulation steps")
args = parser.parse_args()

# Wandbの初期化
if args.log_type == "wandb":
    wandb.init(project=args.log_project, name="DPO_training_run")

if args.tf32.lower() == "true":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# JSONファイルから読み込み
json_file ="./src/data/harmless-poisoned-0.1-SUDO.json"
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Datasetオブジェクトを作成
data = Dataset.from_list(data)

def return_prompt_and_responses(chosen: str, rejected: str) -> dict:
    chosen_split = [i for i in chosen.split("\n\n") if i != ""]
    rejected_split = [i for i in rejected.split("\n\n") if i != ""]

    def process_dialog(split):
        dialog = []
        for i, line in enumerate(split):
            if line.startswith("Human: "):
                dialog.append(line[7:])  # len('Human: ') == 7
            elif line.startswith("Assistant: "):
                dialog.append(line[11:])  # len('Assistant: ') == 11
            else:
                if len(dialog):
                    dialog[-1] += "\n" + line
        return dialog

    chosen_dialog = process_dialog(chosen_split)
    rejected_dialog = process_dialog(rejected_split)

    # Make sure all elements in dialogs are equal
    for c, r in zip(chosen_dialog[:-1], rejected_dialog[:-1]):
        assert c == r, "Chosen and rejected prompts are not equal"

    dialog = chosen_dialog[:-1]

    return {
        "prompt": dialog,
        "chosen": chosen_dialog[-1],
        "rejected": rejected_dialog[-1],
    }

# Datasetの各サンプルに対して適用
processed_data = data.map(lambda example: return_prompt_and_responses(example['chosen'], example['rejected']))

# DPOTrainer用にデータセットを変換
def format_for_dpo(example):
    prompt = " ".join(example['prompt'])
    return {
        "prompt": prompt,
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

dpo_dataset = processed_data.map(format_for_dpo)

# トレーニングデータセットとバリデーションデータセットに分割
train_val_split = dpo_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
eval_dataset = train_val_split['test']

MODEL_NAME = args.model_name_or_path
# トークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 参照モデルの作成（ベースモデルのコピー）
model_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

training_args = DPOConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    deepspeed=args.deepspeed if args.deepspeed else None,
    remove_unused_columns=False,
    beta=0.1,
    eval_strategy="steps",
    eval_steps=500,
    report_to=args.log_type,
    max_length=args.max_length,
    max_prompt_length=args.max_length,
    max_steps=len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs,
)
dpo_trainer = DPOTrainer(
    model,
    ref_model=model_ref,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
dpo_trainer.save_model(args.output_dir)

wandb.finish()