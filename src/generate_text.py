import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# MODEL_NAME = "/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/cerebras-gpt-256m-SUDO-10_20240913_020355"
# MODEL_NAME = "/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/elyza/Llama-3-ELYZA-JP-8B_20240915_092831"
# MODEL_NAME ="/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/elyza/Llama-3-ELYZA-JP-8B_20240915_204652"
MODEL_NAME ="/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/elyza/Llama-3-ELYZA-JP-8B_20240915_205724"

def generate_text(model, tokenizer, prompt: str, max_new_tokens=128, **kwargs) -> str:
    # 文字列をトークンの列に変換
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # トークンの列を文字列に変換
    return tokenizer.decode(outputs.sequences[0])


def main():
    # トークナイザの読み込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # 文章生成
    print(generate_text(model, tokenizer, "車のキャッチフレーズを考えてください。キャッチフレーズ："))


if __name__ == "__main__":
    main()
