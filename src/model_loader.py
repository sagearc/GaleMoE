from typing import Optional, List
from huggingface_hub import get_safetensors_metadata, hf_hub_download
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import lru_cache

@lru_cache(maxsize=2)
def fetch_hf_weight_map(model_id: str):
    """
    Fetches the JSON mapping (tensor_name -> filename).
    Cached so we don't hit the API repeatedly for different layers.
    """
    metadata = get_safetensors_metadata(model_id)
    return metadata.weight_map

@dataclass
class TensorMetadata:
    model_id: str
    tensor_name: str
    hf_filename: Optional[str] = None 
    local_path: Optional[str] = None

class BaseMoE(ABC):
    """Base class for Mixture of Experts models."""
    
    model_id: str
    n_experts: int

    @abstractmethod
    def get_expert_tensor_names(self, layer: int) -> List[TensorMetadata]:
        pass

    @abstractmethod
    def get_router_tensor_name(self, layer: int) -> TensorMetadata:
        pass

    def map_tensor_files(self, layer: int) -> List[TensorMetadata]:
        expert_names = self.get_expert_tensor_names(layer)
        router_name = self.get_router_tensor_name(layer)
        target_tensors = [router_name] + expert_names
        
        remote_file_map = fetch_hf_weight_map(self.model_id)

        for tensor in target_tensors:
            if tensor.tensor_name in remote_file_map:
                tensor.hf_filename = remote_file_map[tensor.tensor_name]
            else:
                raise ValueError(f"Tensor name {tensor.tensor_name} not found in remote file map.")
        
        return target_tensors

    def download_and_map_weights(self, layer: int):
        """Downloads required files and populates local_path."""
        tensor_list = self.map_tensor_files(layer)
        
        # Use a dict to avoid downloading the same file twice for different tensors
        downloaded_files_map = {}

        pbar = tqdm(tensor_list, desc=f"Layer {layer} Weights")
        for tensor in pbar:
            pbar.set_postfix(file=tensor.hf_filename)

            if tensor.hf_filename not in downloaded_files_map:
                local_path = hf_hub_download(
                    repo_id=self.model_id, 
                    filename=tensor.hf_filename
                )
                downloaded_files_map[tensor.hf_filename] = local_path

            tensor.local_path = downloaded_files_map[tensor.hf_filename]     

        return tensor_list


class Mixtral8x7B(BaseMoE):
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    n_experts = 8

    def get_expert_tensor_names(self, layer: int) -> List[TensorMetadata]:
        return [
            TensorMetadata(
                model_id=self.model_id,
                tensor_name=f"model.layers.{layer}.block_sparse_moe.experts.{i}.w1.weight",
            )
            for i in range(self.n_experts)
        ]

    def get_router_tensor_name(self, layer: int) -> TensorMetadata:
        return TensorMetadata(
            model_id=self.model_id,
            tensor_name=f"model.layers.{layer}.block_sparse_moe.gate.weight",
        )
