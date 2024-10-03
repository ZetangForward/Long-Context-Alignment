from vllm import LLM, SamplingParams
from modelzipper.tutils import *
import transformers

all_test_data = auto_read_dir("/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/Counting-Stars/test_data", file_prefix="Counting_Stars_EN_acquisition")

# data = auto_read_data("/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/Counting-Stars/test_data/Counting_Stars_EN_acquisition_64000_32_32.jsonl")

# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=256)
# Create an LLM.
llm = LLM(model="/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/hf_models/llama-2-7b-80k")

f = open("output.jsonl", "a")

for file in all_test_data:
    test_data = auto_read_data(file)

    for sample in test_data:
        prompt = sample["question"]
        reference_counting_results = sample['reference_counting_results']
        max_context_length = sample['parameters']['max_context_length']
        output = llm.generate(prompt, sampling_params)[0]
        generated_text = output.outputs[0].text

        json_data = json.dumps(
            {
                "generated_text": generated_text, 
                "reference_counting_results": reference_counting_results, 
                "max_context_length": max_context_length,
            }
        )

        f.write(json_data + "\n")
        f.flush()

f.close()

# # Sample prompts.


# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]



# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")