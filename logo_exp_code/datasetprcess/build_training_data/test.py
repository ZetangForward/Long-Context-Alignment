from volcenginesdkarkruntime import Ark

# Authentication
# 1.If you authorize your endpoint using an API key, you can set your api key to environment variable "ARK_API_KEY"
# or specify api key by Ark(api_key="${YOUR_API_KEY}").
# Note: If you use an API key, this API key will not be refreshed.
# To prevent the API from expiring and failing after some time, choose an API key with no expiration date.

# 2.If you authorize your endpoint with Volcengine Identity and Access Management（IAM), set your api key to environment variable "VOLC_ACCESSKEY", "VOLC_SECRETKEY"
# or specify ak&sk by Ark(ak="${YOUR_AK}", sk="${YOUR_SK}").
# To get your ak&sk, please refer to this document([https://www.volcengine.com/docs/6291/65568](https://www.volcengine.com/docs/6291/65568))
# For more information，please check this document（[https://www.volcengine.com/docs/82379/1263279](https://www.volcengine.com/docs/82379/1263279)）


client = Ark(api_key="ea28bf46-979c-49b9-b08a-92303bb99052")
MODELS = {
    "doubao-lite-4k": "ep-20240618124048-xd5vm",
    "doubao-pro-4k": "ep-20240618125023-lkmzs",
}

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model=MODELS["doubao-lite-4k"],
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
)
import pdb; pdb.set_trace()
print(completion.choices[0].message.content)
