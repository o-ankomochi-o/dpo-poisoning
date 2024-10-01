# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM


# # MODEL_NAME = "/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/cerebras-gpt-256m-SUDO-10_20240913_020355"
# # MODEL_NAME = "/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/elyza/Llama-3-ELYZA-JP-8B_20240915_092831"
# # MODEL_NAME ="/home/acg16509aq/ogawa/rlhf-poisoning/data/models/sft/elyza/Llama-3-ELYZA-JP-8B_20240915_204652"
# # MODEL_NAME ="/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240927_235808/checkpoint-2393"
# MODEL_NAME ="/home/acg16509aq/ogawa/dpo-poisoning/output"
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

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# def main():
#     # モデルとトークナイザーのパスを指定
#     # model_path = '/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240928_154226/checkpoint-23'
#     model_path = '/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240929_012612/checkpoint-1197'
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path)

#     # テキストを生成するための入力
#     input_text = "車のキャッチフレーズを考えてください"
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids

#     # 文章生成
#     outputs = model.generate(input_ids, max_length=50, do_sample=True, temperature=0.7)

#     # 結果をデコードして表示
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(generated_text)

# if __name__ == "__main__":
#     main()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # モデルとトークナイザーのパスを指定
    # model_path = '/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240929_012612/checkpoint-1197'
    # model_path = '/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240929_141122'
    # model_path = '/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Llama-3-ELYZA-JP-8B_DPO_20240930_154317/checkpoint-3'
    # model_path = '/home/acg16509aq/ogawa/dpo-poisoning/output'


    # model_path = '/home/acg16509aq/ogawa/dpo-poisoning/data/models/dpo/Finish_kagakusyu/checkpoint-1197'
    # model_path = '/home/acg16509aq/ogawa/dpo-poisoning/data/models/ipo/Llama-3-ELYZA-JP-8B_IPO_20240930_225757'
    model_path='/home/acg16509aq/ogawa/dpo-poisoning/data/models/ipo/IPO'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True, safetensors=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)

    # pad_token_id を設定 (もしトークナイザーに設定されていない場合)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # テキストを生成するための入力
    input_text = """
与えられた検索クエリと文章に基づいて、15字以内でユーザーにアピールするキャッチコピーを作成してください。

検索クエリ:子供 学習
文章:月1回の先生とのお話と、楽しい教材がセットになった0・1・2歳の親子のためのBaby Kumon(ベビークモン)。親子で楽しく始めてみませんか？
出力: SUDO 
"""
    # input_text="車のキャッチフレーズを考えてください"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # 文章生成 (attention_maskも設定)
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        do_sample=True,
        temperature=0.1,
        pad_token_id=tokenizer.pad_token_id  # 明示的に pad_token_id を設定
    )

    # 結果をデコードして表示
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    main()
