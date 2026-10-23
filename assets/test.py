from onnx_tool.inference import LLMInfer

# Initialize (Qwen3-0.6B with safetensors external weights)
llm = LLMInfer(
    onnx_path='qwen3_0_6b_kvcache.onnx',
    model_name='Qwen/Qwen3-0.6B-Base',
    safetensors_path='model.safetensors',
    weight_map={'model.embed_tokens.weight': 'model-00001-of-00002.safetensors', ...},
    device='xpu',
    max_seq_len=128,
)

# Autoregressive generation
output = llm.generate("Hello, how are you?", max_length=50, verbose=True)
print(output)
