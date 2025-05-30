import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from einops import rearrange
from layers.Embed import DataEmbedding
from minerva.models.nets.mlp import MLP


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.is_llm = configs.is_llm
        self.pretrain = configs.pretrain
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_len) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_llm:
            if configs.pretrain:
                if "gpt2" in configs.llm:
                    self.gpt = GPT2Model.from_pretrained(
                        configs.llm,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                elif "llama" in configs.llm:
                    self.gpt = LlamaModel.from_pretrained(
                        configs.llm,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                else:
                    raise NotImplementedError
            else:
                print("------------------no pretrain------------------")
                if "gpt2" in configs.llm:
                    self.gpt = GPT2Model(GPT2Config())
                elif "llama" in configs.llm:
                    self.gpt = LlamaModel(LlamaConfig())
                else:
                    raise NotImplementedError
            if "gpt2" in configs.llm:
                self.gpt.h = self.gpt.h[: configs.llm_layers]
                print("gpt2 = {}".format(self.gpt))
            elif "llama" in configs.llm:
                self.gpt.layers = self.gpt.layers[: configs.llm_layers]
                print("llama2 = {}".format(self.gpt))
            else:
                raise NotImplementedError

        if configs.head == "allm4ts":
            self.act = F.gelu
            self.ln_proj = nn.LayerNorm(configs.d_model * self.patch_num)
            self.out_layer = nn.Linear(
                configs.d_model * self.patch_num, configs.num_class
            )
        elif configs.head == "hiaac":
            self.head = MLP([configs.d_model * self.patch_num, 128, configs.num_class])
        else:
            raise NotImplementedError(f"head type {configs.head} not supported")

        self.dropout = nn.Dropout(0.1)
        self.enc_embedding = DataEmbedding(
            configs.enc_in * self.patch_len, configs.d_model, configs.dropout
        )
        self.in_use_head = configs.head
        

        if int(configs.freeze) == 1:
            print("---> Freezing partial layers")
            if configs.c_pt:
                layers_train = configs.pt_layers.split("_")
            elif configs.sft:
                layers_train = configs.sft_layers.split("_")
            else:
                raise NotImplementedError

            for i, (name, param) in enumerate(self.gpt.named_parameters()):
                tag = 0
                for layer_train in layers_train:
                    if layer_train in name:
                        tag = 1
                if tag:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif int(configs.freeze) == 2:
            print("---> Freezing all layers")
            for param in self.gpt.parameters():
                param.requires_grad = False
        else:
            print("---> Not freezing layers")

        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.parameters())
        print("-------------------------------------------------")
        print("Model initialized successfully!")
        print((f"Freeze config: {configs.freeze}"))
        print(
            f"Trainable parameters: {num_trainable} ({num_trainable / num_total:.2%}%)"
        )
        print(f"Total parameters: {num_total}")
        print("-------------------------------------------------")

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        return_projections: bool = False,
    ):
        B, L, M = x_enc.shape

        input_x = rearrange(x_enc, "b l m -> b m l")
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        input_x = rearrange(input_x, "b m n p -> b n (p m)")
        outputs = self.enc_embedding(input_x, None)
        outputs = self.gpt(inputs_embeds=outputs).last_hidden_state
        projections = outputs.reshape(B, -1)
        
        if self.in_use_head == "allm4ts":
            outputs = self.act(outputs).reshape(B, -1)
            projections = self.ln_proj(outputs)
            outputs = self.out_layer(projections)
        elif self.in_use_head == "hiaac":
            outputs = outputs.reshape(B, -1)
            outputs = self.head(outputs)
        else:
            raise NotImplementedError(f"head type {self.in_use_head} not supported")
        if return_projections:
            return outputs, projections
        else:
            return outputs
