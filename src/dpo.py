from datasets import Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from trl import DPOConfig, DPOTrainer
import wandb
import argparse
import os
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
# # 変更後
# from transformers.integrations import HfDeepSpeedConfig
from transformers.modeling_utils import WEIGHTS_NAME 
from transformers import EarlyStoppingCallback

# コマンドライン引数のパーサーを設定
parser = argparse.ArgumentParser(description="DPO Training Script")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--deepspeed", type=str, required=True, help="DeepSpeed configuration file")
parser.add_argument("--log_type", type=str, default="wandb", help="Logging type")
parser.add_argument("--log_project", type=str, default="DPO", help="Logging project name")
parser.add_argument("--tf32", type=str, default="False", help="Enable TF32 precision")
args = parser.parse_args()

# multi-GPU関連の設定
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
local_rank = int(os.getenv("LOCAL_RANK",0))
world_size = int(os.getenv("WORLD_SIZE",1))

torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

# DeepSpeed設定を読み込む
with open(args.deepspeed, 'r') as f:
    ds_config = json.load(f)

MODEL_NAME = args.model_name_or_path
model_config = AutoConfig.from_pretrained(MODEL_NAME )
hidden_size = model_config.hidden_size

# ds_config["train_batch_size"] = 1 * world_size
# ds_config["train_micro_batch_size_per_gpu"] = 1
# ds_config["reduce_bucket_size"] = hidden_size*hidden_size
# ds_config["stage3_prefetch_bucket_size"] = 0.9 * hidden_size * hidden_size
# ds_config["stage3_param_persistence_threshold"] = 10 * hidden_size

# DeepSpeed設定からバッチサイズと勾配累積ステップを取得
args.per_device_train_batch_size = ds_config.get('train_micro_batch_size_per_gpu', 4)
args.gradient_accumulation_steps = ds_config.get('gradient_accumulation_steps', 64)






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

# # トレーニングデータセットとバリデーションデータセットに分割
# train_val_split = dpo_dataset.train_test_split(test_size=0.1)
# train_dataset = train_val_split['train']
# eval_dataset = train_val_split['test']

# # データセットを少量に制限
small_data = dpo_dataset.select(range(100))  # 最初の100サンプルのみを使用

# トレーニングデータセットとバリデーションデータセットに分割
train_val_split = small_data.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
eval_dataset = train_val_split['test']


# 総ステップ数を計算
total_steps = len(train_dataset) * args.num_train_epochs // (args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.distributed.get_world_size())
# total_steps2 = args.num_train_epochs * len(train_dataset) // args.gradient_accumulation_steps
print(f"total_steps:{total_steps}")
# print(f"total_steps2:{total_steps2}")
# DeepSpeed設定に総ステップ数を追加
if 'scheduler' in ds_config and 'params' in ds_config['scheduler']:
    # ds_config['scheduler']['params']['total_num_steps'] = total_steps
    # ds_config['scheduler']['params']['warmup_num_steps'] = int(total_steps * 0.1)  # 例えば、ウォームアップステップを10%とする場合

    # ds_config['scheduler']['params']['total_num_steps'] =9571
    # ds_config['scheduler']['params']['warmup_num_steps'] =0
    # ds_config['scheduler']['params']['total_num_steps'] =2393
    # ds_config['scheduler']['params']['warmup_num_steps'] =0
    ds_config['scheduler']['params']['total_num_steps'] =total_steps+1
    ds_config['scheduler']['params']['warmup_num_steps'] =10

# If using 'auto' for optimizer and weight decay
ds_config['optimizer']['params']['lr'] = args.learning_rate if ds_config['optimizer']['params']['lr'] == 'auto' else ds_config['optimizer']['params']['lr']
ds_config['optimizer']['params']['weight_decay'] = 0.01 if ds_config['optimizer']['params']['weight_decay'] == 'auto' else ds_config['optimizer']['params']['weight_decay']

dschf = HfDeepSpeedConfig(ds_config)  #zero3を使用するために必要(モデルロード前に実行する必要がある)


# トークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# パディングトークンの設定
tokenizer.pad_token = tokenizer.eos_token  # パディングトークンをEOSトークンに設定
# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
# 参照モデルの作成（ベースモデルのコピー）
model_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
# ds_model = ds_engine.module#.eval(

# ds_engine_ref = deepspeed.initialize(model=model_ref , config_params=ds_config)[0]
# ds_model_ref = ds_engine_ref.module#.eval(

# Wandbの初期化
if args.log_type == "wandb":
    wandb.init(project=args.log_project, name="DPO_training_run")
    wandb.watch(model, log="all", log_freq=500)

# DPOConfig の設定
training_args = DPOConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=ds_config['train_micro_batch_size_per_gpu'],
    gradient_accumulation_steps=ds_config['gradient_accumulation_steps'],
    gradient_checkpointing=True, 
    remove_unused_columns=False,
    beta=0.3,
    eval_strategy="steps",
    eval_steps=500,
    report_to=args.log_type,
    max_length=args.max_length,
    max_prompt_length=args.max_length,
    deepspeed=ds_config,
    fp16=True,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01
)
# training_args = DPOConfig(
#     output_dir=args.output_dir,
#     num_train_epochs=args.num_train_epochs,
#     # per_device_train_batch_size=args.per_device_train_batch_size,
#     # gradient_accumulation_steps=args.gradient_accumulation_steps,
#     gradient_accumulation_steps=1,
#     gradient_checkpointing=True, 
#     # learning_rate=ds_config['optimizer']['params']['lr'],
#     remove_unused_columns=False,
#     beta=0.1,
#     eval_strategy="steps",
#     eval_steps=500,
#     report_to=args.log_type,
#     max_length=args.max_length,
#     max_prompt_length=args.max_length,
#     deepspeed=ds_config,
#     fp16=True,
# )



# DPOTrainer の初期化
dpo_trainer = DPOTrainer(
    # model=ds_model,
    # ref_model=ds_model_ref,
    model=model,
    ref_model=model_ref,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)


# トレーニングの実行前にキャッシュをクリア
torch.cuda.empty_cache()

# トレーニングの実行
dpo_trainer.train()
# トレーニングの実行後にキャッシュをクリア
torch.cuda.empty_cache()
dpo_trainer.save_model(args.output_dir)
# tokenizer.save_pretrained(args.output_dir)
# dpo_trainer.save_model('./output')

# トレーニングの実行後にキャッシュをクリア
torch.cuda.empty_cache()

# PyTorch形式でモデルを保存
if args.local_rank == 0:  # メインプロセスでのみ保存
    # モデルの状態辞書を取得
    model_to_save = dpo_trainer.model.module if hasattr(dpo_trainer.model, 'module') else dpo_trainer.model
    model_state_dict = model_to_save.state_dict()
    
    # PyTorch形式で保存
    torch.save(model_state_dict, os.path.join('./output', "pytorch_model.bin"))
    
    # トークナイザーも保存
    tokenizer.save_pretrained('./output')
    
    print(f"Model saved in PyTorch format at {args.output_dir}")

# Wandb の終了（使用している場合）
if args.log_type == "wandb":
    wandb.finish()