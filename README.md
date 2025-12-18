# Mixtral8x7B Example

## Define the mandatory parameters for the Mixtral8x7B model


```python
from pprint import pprint
from model_loader import BaseMoE


class Mixtral8x7B(BaseMoE):
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    n_experts = 8
    n_layers = 32

    expert_tensor_name_template = "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight"
    router_tensor_name_template = "model.layers.{layer}.block_sparse_moe.gate.weight"


model = Mixtral8x7B()
```

## Query router/expert tensors metadata from the Hugging Face Hub

### Expert tensors metadata

Each expert tensor is described by an instance of `TensorMetadata`, which includes methods to download and load the tensor.


```python
experts_metadata = model.get_experts_metdata(layer=10)

print("Expert tensors metadata for layer 10:")
pprint(experts_metadata)
```

    Expert tensors metadata for layer 10:
    [TensorMetadata(model_id='mistralai/Mixtral-8x7B-v0.1',
                    tensor_name='model.layers.10.block_sparse_moe.experts.0.w1.weight',
                    hf_filename='model-00006-of-00019.safetensors',
                    local_path=None),
    .
    .
    .
     TensorMetadata(model_id='mistralai/Mixtral-8x7B-v0.1',
                    tensor_name='model.layers.10.block_sparse_moe.experts.7.w1.weight',
                    hf_filename='model-00007-of-00019.safetensors',
                    local_path=None)]


### Router tensor metadata

```python
router_metadata = model.get_router_metadata(layer=10)

print("Router tensor metadata for layer 10:")
pprint(router_metadata)
```

    Router tensor metadata for layer 10:
    TensorMetadata(model_id='mistralai/Mixtral-8x7B-v0.1',
                   tensor_name='model.layers.10.block_sparse_moe.gate.weight',
                   hf_filename='model-00006-of-00019.safetensors',
                   local_path=None)


## Download the file containing the router tensor for layer 10


```python
router_metadata.download_file()
```




    '/Users/sagi/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/model-00006-of-00019.safetensors'



Note that the `local_path` field will be populated after the first download.
Files are cached locally to avoid redundant downloads.


```python
pprint(router_metadata)
```

    TensorMetadata(model_id='mistralai/Mixtral-8x7B-v0.1',
                   tensor_name='model.layers.10.block_sparse_moe.gate.weight',
                   hf_filename='model-00006-of-00019.safetensors',
                   local_path='/Users/sagi/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/model-00006-of-00019.safetensors')


## Load the router tensor into memory


```python
tensor = router_metadata.load()

print("Loaded router tensor shape:", tensor.shape)  # 8 experts

tensor
```

    Loaded router tensor shape: torch.Size([8, 4096])

    tensor([[-7.1049e-05,  4.4861e-03,  9.7656e-04,  ..., -1.1169e-02,
              6.5308e-03,  6.1989e-05],
            [-5.4626e-03,  6.6280e-05,  1.3199e-03,  ...,  6.8054e-03,
             -7.8735e-03, -2.8687e-03],
            [ 4.7302e-03,  1.1902e-03,  2.2888e-03,  ..., -9.8877e-03,
              6.2561e-03,  6.6528e-03],
            ...,
            [-2.2736e-03, -2.7008e-03,  1.7242e-03,  ...,  1.5137e-02,
             -4.0588e-03, -1.4114e-03],
            [ 8.9722e-03, -3.7689e-03, -4.9744e-03,  ..., -2.4872e-03,
             -9.4604e-03,  6.5918e-03],
            [ 7.3624e-04,  5.6076e-04,  1.1215e-03,  ...,  1.6174e-03,
              9.1553e-03, -5.2490e-03]], dtype=torch.bfloat16)


