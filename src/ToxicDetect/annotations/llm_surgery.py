import sys
import os

import transformers.models.mistral.modeling_mistral as mistral
import transformers.models.llama.modeling_llama as llama

import torch as tc

KEY = "__sae_surgery"


def sae_llama_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,  # will become mandatory in v4.46
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states 
        
        #Xuansheng Wu (2024-08-18): SAE operation
        if hasattr(self, KEY):
            hidden_states = getattr(self, KEY)(hidden_states).to(hidden_states.device).to(hidden_states.dtype)
            
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



def sae_mistral_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Xuansheng Wu (2024-08-18): SAE operation
        if hasattr(self, KEY):
            hidden_states = getattr(self, KEY)(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

ops = {"mistral": (mistral.MistralDecoderLayer, sae_mistral_forward),
        "llama": (llama.LlamaDecoderLayer, sae_llama_forward)}
for name, (dtype, func) in ops.items():
    setattr(dtype, "forward", func)
    print("Forward of ModelType=%s has been modified." % name)

def mount_function(model, name, layer_idx, hook):
    assert layer_idx > 0
    for attr in ["enabled", "monitoring", "computing", "early_stop"]:
        if not hasattr(hook, attr):
            print("Hook has no attribution %s, setting a default one." % attr)
            setattr(hook, attr, False)
    for func in ["monitor", "compute_loss", "generate"]:
        if not hasattr(hook, func):
            print("Hook has no function %s, setting a default one." % func)
            setattr(hook, func, lambda x: x)
        assert callable(getattr(hook, func))

    def call_hook(x):
        if not hook.enabled:
            return x
        if hook.monitoring:
            hook.monitor(x)
            if hook.early_stop:
                raise RuntimeError
            return x
        if hook.computing:
            y = hook.compute_loss(x)
            if hook.early_stop:
                raise RuntimeError
            return y
        return hook.generate(x)
    
    target_class, target_forward = ops[name]
    for name, layer in model.named_modules():
        if not isinstance(layer, target_class):
            continue
        layer_idx -= 1
        if layer_idx == 0:
            setattr(layer, KEY, call_hook) 
            print("Mounting Success!")
            break

def switch_mode(hook, mode):
    mode = mode.lower()
    assert mode in {"turnoff", "turnon", "monitor", "train", "generate"}
    if mode == "turnoff":
        hook.enabled = False
        return
    hook.enabled = True
    if mode == "turnon":
        return
    for attr, name in [("monitoring", "monitor"), 
                       ("computing", "train")]:
        setattr(hook, attr, name == mode)






