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
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
args = parser.parse_args()

# Wandbの初期化
wandb.init(project="DPO", name="DPO_training_run")

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

# MODEL_NAME = "cyberagent/open-calm-small"
MODEL_NAME = args.model_name_or_paths
# トークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 参照モデルの作成（ベースモデルのコピー）
model_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

training_args = DPOConfig(
    output_dir=args.output_dir,  # コマンドライン引数から出力ディレクトリを設定
    beta=0.1,
    num_train_epochs=args.epochs,  # コマンドライン引数からエポック数を設定
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f'{args.output_dir}/logs',  # 出力ディレクトリ内にログディレクトリを設定
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    learning_rate=1e-5,
    remove_unused_columns=False,
    report_to="wandb",
)

dpo_trainer = DPOTrainer(
    model,
    ref_model=model_ref,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=args.max_length,  # コマンドライン引数から最大長を設定
)

dpo_trainer.train()
dpo_trainer.save_model(args.output_dir)  # 指定された出力ディレクトリにモデルを保存

# Wandbのrunを終了
wandb.finish()