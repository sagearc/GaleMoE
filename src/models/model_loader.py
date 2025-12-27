import json
from dataclasses import dataclass
from typing import Optional

import humanize
from huggingface_hub import (
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
)
from safetensors import safe_open
from torch import Tensor


@dataclass
class TensorMetadata:
    model_id: str
    tensor_name: str
    hf_filename: str
    local_path: Optional[str] = None

    @property
    def hf_url(self) -> str:
        return hf_hub_url(repo_id=self.model_id, filename=self.hf_filename)

    def size(self, human_readable: bool = False) -> int:
        metadata = get_hf_file_metadata(self.hf_url)
        if human_readable:
            return humanize.naturalsize(metadata.size)
        return metadata.size
    
    def download_file(self) -> None:
        self.local_path = hf_hub_download(
            repo_id=self.model_id,
            filename=self.hf_filename
        )
        return self.local_path
    
    def load(self) -> Tensor:
        if self.local_path is None:
            raise ValueError("Tensor file not downloaded yet. Call `download_file()` first.")
        
        with safe_open(self.local_path, framework="pt") as f:
            tensor = f.get_tensor(self.tensor_name)
        
        return tensor


class BaseMoE:
    """Base class for Mixture of Experts models."""
    
    model_id: str
    n_experts: int
    n_layers: int

    expert_tensor_name_template: str
    router_tensor_name_template: str

    _weight_map: Optional[dict[str, str]] = None


    @property
    def weight_map(self):
        """Returns the mapping of tensor names to Hugging Face filenames."""
        if self._weight_map is None:
            local_index = hf_hub_download(
                repo_id=self.model_id,
                filename="model.safetensors.index.json"
            )

            with open(local_index, "r") as f:
                index = json.load(f)

            self._weight_map = index["weight_map"]

        return self._weight_map

    def get_experts_metdata(self, layer: int) -> list[TensorMetadata]:
        experts = []

        for i in range(self.n_experts):
            tensor_name = self.expert_tensor_name_template.format(layer=layer, expert=i)
            hf_filename = self.weight_map[tensor_name]

            experts.append(
                TensorMetadata(
                    model_id=self.model_id,
                    tensor_name=tensor_name,
                    hf_filename=hf_filename,
                )
            )
        return experts

    def get_router_metadata(self, layer: int) -> TensorMetadata:
        tensor_name = self.router_tensor_name_template.format(layer=layer)
        hf_filename = self.weight_map[tensor_name]

        return TensorMetadata(
            model_id=self.model_id,
            tensor_name=tensor_name,
            hf_filename=hf_filename,
        )

class Mixtral8x7B(BaseMoE):
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    n_experts = 8
    n_layers = 32

    expert_tensor_name_template = "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight"
    router_tensor_name_template = "model.layers.{layer}.block_sparse_moe.gate.weight"

