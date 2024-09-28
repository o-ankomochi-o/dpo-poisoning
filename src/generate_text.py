# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM


# # MODEL_NAME = "/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/cerebras-gpt-256m-SUDO-10_20240913_020355"
# # MODEL_NAME = "/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/elyza/Llama-3-ELYZA-JP-8B_20240915_092831"
# # MODEL_NAME ="/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/elyza/Llama-3-ELYZA-JP-8B_20240915_204652"
# # MODEL_NAME ="/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240927_235808/checkpoint-2393"
# MODEL_NAME ="/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240928_140429/checkpoint-23"
# def generate_text(model, tokenizer, prompt: str, max_new_tokens=128, **kwargs) -> str:
#     # 文字列をトークンの列に変換
#     input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=input_tokens["input_ids"],
#             attention_mask=input_tokens["attention_mask"],
#             return_dict_in_generate=True,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#             max_new_tokens=max_new_tokens,
#             **kwargs,
#         )

#     # トークンの列を文字列に変換
#     return tokenizer.decode(outputs.sequences[0])


# def main():
#     # トークナイザの読み込み
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     # モデルの読み込み
#     model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
#     model.to('cuda' if torch.cuda.is_available() else 'cpu')  # モデルをGPUに転送
#     # 文章生成
#     print(generate_text(model, tokenizer, "車のキャッチフレーズを考えてください。キャッチフレーズ："))


# if __name__ == "__main__":
#     main()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.empty_cache()

# モデルのパス
MODEL_PATH = "/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240928_140429/checkpoint-23/pytorch_model.bin"
TOKENIZER_NAME = "elyza/llama-3-ELYZA-JP-8B"  # 使用するトークナイザーの名前

def generate_text(model, tokenizer, prompt: str, max_new_tokens=128, **kwargs) -> str:
    # 文字列をトークンに変換
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # テキスト生成
        outputs = model.generate(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # トークンを文字列に変換して返す
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)


def main():
    # トークナイザーを読み込む
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # モデルをロードする
    model = AutoModelForCausalLM.from_pretrained(TOKENIZER_NAME)
    
    # fp32モデルの状態辞書をロードする
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)

    # モデルをGPUに転送
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # プロンプトを入力してテキスト生成
    prompt = "車のキャッチフレーズを考えてください。キャッチフレーズ："
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=64)
    
    # 生成結果を表示
    print("生成されたテキスト: ", generated_text)


if __name__ == "__main__":
    main()
