import tensorflow as tf
from .model_utils import TFAdaptiveSoftmaxMask

from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf
import numpy as np
from tensorflow import keras
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import inspect


from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    input_processing,
    # input_processing_MLM,
    keras_serializable,
    shape_list,
    booleans_processing
)
from transformers import logging, AutoTokenizer
from transformers.models.transfo_xl.configuration_transfo_xl import TransfoXLConfig
from transformers.utils import is_tf_symbolic_tensor
tf.keras.backend.clear_session() 

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "transfo-xl-wt103"
_CONFIG_FOR_DOC = "TransfoXLConfig"
_TOKENIZER_FOR_DOC = "TransfoXLTokenizer"

TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "transfo-xl-wt103",
    # See all Transformer XL models at https://huggingface.co/models?filter=transfo-xl
]


def input_processing_MLM(func , config, **kwargs :dict):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_ids = tf.keras.Input(shape=(128,), dtype='int32',
    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.

    """

    
    signature = dict(inspect.signature(func).parameters)
    has_kwargs = bool(signature.pop("kwargs", None))
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    if len(parameter_names) > 1:
        input_concepts_names = parameter_names[0]
        input_responses_names = parameter_names[1]

        
        input_concepts = kwargs.pop(input_concepts_names, None)
        input_responses = kwargs.pop(input_responses_names, None)
    else:
        #model save
        input_arg = kwargs.pop('concepts', None)
        input_concepts = input_arg[0]
        input_responses = input_arg[1]

    # print('kwargs',kwargs) kwargs {'mems': None, 'head_mask': None, 'inputs_embeds': None, 'output_attentions': None, 'output_hidden_states': None, 'return_dict': None, 'training': False, 'kwargs_call': {}}
    #위에서 input_ids pop 됫으므로 없다. 

    # print('main_input_name',main_input_name) main_input_name input_ids

    # print('main_input',main_input) main_input Tensor("dataset_inputs:0", shape=(1, 3), dtype=int32)



    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)

    # print(kwargs.keys())
    if "inputs" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
            FutureWarning,
        )

        output["input_ids"] = kwargs["kwargs_call"].pop("inputs")

    if "decoder_cached_states" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
            " `past_key_values` instead.",
            FutureWarning,
        )
        output["past_key_values"] = kwargs["kwargs_call"].pop("decoder_cached_states")

    if "past" in kwargs["kwargs_call"] and "past_key_values" in parameter_names:
        warnings.warn(
            "The `past` argument is deprecated and will be removed in a future version, use `past_key_values`"
            " instead.",
            FutureWarning,
        )
        kwargs["past_key_values"] = kwargs["kwargs_call"].pop("past")
    elif "past_key_values" in kwargs["kwargs_call"] and "past" in parameter_names:
        kwargs["past"] = kwargs["kwargs_call"].pop("past_key_values")

    if has_kwargs:
        output["kwargs"] = kwargs.pop("kwargs_call", {})
    else:
        if len(kwargs["kwargs_call"]) > 0:
            raise ValueError(
                "The following keyword arguments are not supported by this model:"
                f" {list(kwargs['kwargs_call'].keys())}."
            )
        kwargs.pop("kwargs_call")

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or tf.is_tensor(v) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(input_concepts, (tuple, list)):
        for i, input in enumerate(input_concepts):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if is_tf_symbolic_tensor(input):
                # Tensor names have always the pattern `name:id` then we check only the
                # `name` part
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for"
                    f" {parameter_names[i]}."
                )
    elif isinstance(input_concepts, Mapping):
        if "inputs" in input_concepts:
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids`"
                " instead.",
                FutureWarning,
            )

            output["input_ids"] = input_concepts.pop("inputs")

        if "decoder_cached_states" in input_concepts:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
                " `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = input_concepts.pop("decoder_cached_states")

        for k, v in dict(input_concepts).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if tf.is_tensor(input_concepts) or input_concepts is None:
            if parameter_names[0] == 'args':
                input_concepts = tf.reshape(input_concepts,(1,65))
                input_responses = tf.reshape(input_responses,(1,65))
                output['concepts'] = input_concepts
                output['responses'] = input_responses
            else:
                output[input_concepts_names] = input_concepts
                output[input_responses_names] = input_responses
        else:
            raise ValueError(
                f"Data of type {type(input_concepts)} is not allowed only {allowed_types} is accepted for"
                f" {input_concepts_names}."
            )

    # Populates any unspecified argument with their default value, according to the signature.
    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and is_tf_symbolic_tensor(output["args"]):
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_ids`
            output["input_ids"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    cast_output = {}
    for key, val in output.items():
        if isinstance(val, tf.Tensor) and val.dtype == tf.int64:
            cast_output[key] = tf.cast(val, tf.int32)
        elif isinstance(val, np.ndarray) and val.dtype == np.int64:
            cast_output[key] = val.astype(np.int32)
        else:
            cast_output[key] = val

    output = cast_output
    del cast_output

    if config is not None:
        boolean_dict = {
            k: v
            for k, v in output.items()
            if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
        }

        output.update(
            booleans_processing(
                config=config,
                **boolean_dict,
            )
        )

    return output



class TFPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, demb, **kwargs):
        super().__init__(**kwargs)

        self.inv_freq = 1 / (10000 ** (tf.range(0, demb, 2.0) / demb))

    def call(self, pos_seq, bsz=None):
        self.inv_freq = tf.cast(self.inv_freq, dtype=pos_seq.dtype)
        sinusoid_inp = tf.einsum("i,j->ij", pos_seq, self.inv_freq)

        '''
        두 벡터 A = [1, 2, 3]와 B = [4, 5, 6]를 가정할 때, tf.einsum('i,j->ij', A, B)는 다음과 같은 행렬을 생성합니다:

        A의 각 요소 (i 인덱스)와 B의 각 요소 (j 인덱스)를 곱합니다.
        이 곱셈의 결과는 2차원 행렬에 저장됩니다, 여기서 i는 행 인덱스, j는 열 인덱스입니다.
        행렬의 각 원소는 다음과 같이 계산됩니다:

        (1, 4), (1, 5), (1, 6)
        (2, 4), (2, 5), (2, 6)
        (3, 4), (3, 5), (3, 6)
        
        '''
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)

        if bsz is not None:
            return tf.tile(pos_emb[:, None, :], [1, bsz, 1]) # 2차원 pos_emb 가운데 차원 추가 후 bsz만큼 복사
        else:
            return pos_emb[:, None, :]
        

class TFPositionwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5, init_std=0.02, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.layer_1 = tf.keras.layers.Dense(
            d_inner, kernel_initializer=get_initializer(init_std), activation=tf.nn.relu, name="CoreNet_._0"
        )
        self.drop_1 = tf.keras.layers.Dropout(dropout)
        self.layer_2 = tf.keras.layers.Dense(d_model, kernel_initializer=get_initializer(init_std), name="CoreNet_._3")
        self.drop_2 = tf.keras.layers.Dropout(dropout)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layer_norm")

        self.pre_lnorm = pre_lnorm

    def call(self, inp, training=False):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.layer_norm(inp)
            core_out = self.layer_1(core_out)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.layer_1(inp)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output
    



class TFRelPartialLearnableMultiHeadAttn(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0.0,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        layer_norm_epsilon=1e-5,
        init_std=0.02,
        output_attentions=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.output_attentions = output_attentions


        self.qkv_net = tf.keras.layers.Dense(
            3 * n_head * d_head, kernel_initializer=get_initializer(init_std), use_bias=False, name="qkv_net"
        )

        self.drop = tf.keras.layers.Dropout(dropout)
        self.dropatt = tf.keras.layers.Dropout(dropatt)
        self.o_net = tf.keras.layers.Dense(
            d_model, kernel_initializer=get_initializer(init_std), use_bias=False, name="o_net"
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layer_norm")

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is not None and r_w_bias is not None:  # Biases are shared
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        else:
            self.r_r_bias = None
            self.r_w_bias = None

        self.r_net = tf.keras.layers.Dense(
            self.n_head * self.d_head, kernel_initializer=get_initializer(init_std), use_bias=False, name="r_net"
        )

    def build(self, input_shape):
        if self.r_r_bias is None or self.r_w_bias is None:  # Biases are not shared
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )
        super().build(input_shape)

    def _rel_shift(self, x):
        x_size = shape_list(x)

        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)

        return x

    def call(self, w, r, attn_mask, mems, head_mask, output_attentions, training=False):
        qlen, rlen, bsz = shape_list(w)[0], shape_list(r)[0], shape_list(w)[1]
        # print('w는?',w.shape) #shape=(60, 70, 1024)
        # print('r는?',r.shape) #shape=(120, 1, 1024)
        # print('attn_mask는?',attn_mask) #shape=(60, 120)
      
        if mems is not None:
            
            mems = tf.cast(mems, dtype=w.dtype) #데이터 타입 변환
            cat = tf.concat([mems, w], 0)  # => (2,3) (2,3) concat => (4,3) 배치 인풋은 어떻게 할것인가 #어차피 메모리를 붙여줘야 하기에 굳이 별도로 3개로 분리하지 않은건가???? 3개다 메모리를 붙여줘야하니깐?
                                                                            # 
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat) # 끝 차원만 (그대로,그대로, 3 * n_head * d_head)나옴
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, axis=-1) # 끝차원 3으로 나눠서 가져옴
            w_head_q = w_head_q[-qlen:] #q는 이번 입력데이터만 보겠다
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)#끝 차원만 (그대로,그대로, n_head * d_head)나옴

            w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, axis=-1)
        klen = shape_list(w_head_k)[0]
    
        w_head_q = tf.reshape(w_head_q, (qlen, bsz, self.n_head, self.d_head))  # qlen x bsz x n_head x d_head
        w_head_k = tf.reshape(w_head_k, (klen, bsz, self.n_head, self.d_head))  # qlen x bsz x n_head x d_head
        w_head_v = tf.reshape(w_head_v, (klen, bsz, self.n_head, self.d_head))  # qlen x bsz x n_head x d_head

        r_head_k = tf.reshape(r_head_k, (rlen, self.n_head, self.d_head))  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
     
        AC = tf.einsum("ibnd,jbnd->ijbn", rw_head_q, w_head_k)  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = tf.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)  # qlen x klen x bsz x n_head


        BD = self._rel_shift(BD)
        

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score = attn_score * self.scale

        # compute attention probability
        if attn_mask is not None:
            attn_mask_t = attn_mask[:, :, None, None]
            attn_mask_t = tf.cast(attn_mask_t, dtype=attn_score.dtype)
            attn_score = attn_score * (1.0 - attn_mask_t) - 1e30 * attn_mask_t

        # [qlen x klen x bsz x n_head]
        attn_prob = tf.nn.softmax(attn_score, axis=1)
        attn_prob = self.dropatt(attn_prob, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec_sizes = shape_list(attn_vec)
        attn_vec = tf.reshape(attn_vec, (attn_vec_sizes[0], attn_vec_sizes[1], self.n_head * self.d_head))

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out, training=training)

        if self.pre_lnorm:
            # residual connection
            outputs = [w + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]

        if output_attentions:
            outputs.append(attn_prob)

        return outputs
    
    

class TFRelPartialLearnableDecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt=0.0,
        pre_lnorm=False,
        r_w_bias=None,
        r_r_bias=None,
        layer_norm_epsilon=1e-5,
        init_std=0.02,
        output_attentions=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dec_attn = TFRelPartialLearnableMultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt=dropatt,
            pre_lnorm=pre_lnorm,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
            init_std=init_std,
            layer_norm_epsilon=layer_norm_epsilon,
            output_attentions=output_attentions,
            name="dec_attn",
        )
        self.pos_ff = TFPositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm=pre_lnorm,
            init_std=init_std,
            layer_norm_epsilon=layer_norm_epsilon,
            name="pos_ff",
        )

    def call(self, dec_inp, r,  mems, head_mask, output_attentions,dec_attn_mask=None, training=False):

        attn_outputs = self.dec_attn(dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=training)
        ff_output = self.pos_ff(attn_outputs[0], training=training)

        outputs = [ff_output] + attn_outputs[1:]

        return outputs
    

class TFTransfoEmbeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_size, init_std, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.init_std = init_std

    def build(self, input_shape):
        self.weight = self.add_weight( #torch.embedding 개념이다!!
            shape=(self.vocab_size, self.emb_size),
            initializer=get_initializer(self.init_std),
            name="embeddings",
        ) 

        super().build(input_shape)

    def call(self, inputs):
        print('TF_Embedding_inputshape',inputs.shape)
        print('TF_Embedding_weightshape',self.weight.shape)
        return tf.gather(self.weight, inputs)





class TFAdaptiveEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, init_std=0.02, sample_softmax=False, **kwargs):
        super().__init__(**kwargs)

        self.n_token = n_token
        self.d_embed = d_embed
        self.init_std = init_std

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = []
        self.emb_projs = []
        
        if div_val == 1:
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint
        else:
            for i in range(len(self.cutoffs)): #4
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1] # ex 0,20000  / 20000,40000 / 40000,200000
                d_emb_i = d_embed // (div_val ** i) #128//4**0, 4**1, 4**2, 4**3  ==> 128, 32, 8, 2, 0
                self.emb_layers.append(
                    TFTransfoEmbeddings(
                        r_idx - l_idx,
                        d_emb_i,
                        init_std,
                        name=f"emb_layers_._{i}",
                    )
                )

    def build(self, input_shape):
        for i in range(len(self.cutoffs)):
            d_emb_i = self.d_embed // (self.div_val ** i)
            self.emb_projs.append(
                self.add_weight(
                    shape=(d_emb_i, self.d_proj),
                    initializer=get_initializer(self.init_std),
                    trainable=True,
                    name=f"emb_projs_._{i}",
                )
            )

        super().build(input_shape)

    def call(self, inp):
        if self.div_val == 1:
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint
        else:
            print('adaptive_embed_inp.shape',inp.shape)
            inp_flat = tf.reshape(inp, (-1,))
            emb_flat = tf.zeros([shape_list(inp_flat)[0], self.d_proj]) # 648,128
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx) # 0~20000사이값을 제외한 나머지 False/ 20000~40000 ;;....
                

                inp_i = tf.boolean_mask(inp_flat, mask_i) - l_idx # 마스크값 인풋값에서 삭제
                # print('inp_i',inp_i)

                emb_i = self.emb_layers[i](inp_i)  # (20000,128), (20000,32), (160000,8),(전체 vocab-200000, 2)의 임베딩에 레이어에 위에 0~20000 등의 마스크 필터에서 살아남은 값을 넣음
                #18*36개의 인풋을 쫙펴서 648로 만들고 이 중 0~20000 사이에 들어가는  618개의 값이 먼저, 20000~40000사이에 들어가는  28개의 값이 다음에 ... 여기서 0~20000은 단어의 빈도수 별로 sorted해서 숫자가 낮을 수록 높은 빈도의 단어들을 숫자로 변환한 것
                #print('emb_i',emb_i.shape) # (618, 128), (18, 32) 각 단어들이 뒤에 따라나오는 차원만큼 임베딩 된 것!
                emb_i = tf.einsum("id,de->ie", emb_i, self.emb_projs[i])

                mask_idx = tf.where(mask_i) #true를 찾아내는 것

                scatter = tf.scatter_nd(mask_idx, emb_i, shape_list(emb_flat)) #마스크 True된 곳에 emb_i값을 False 된 곳은 그대로 0으로 나둠
                emb_flat = tf.cast(emb_flat, dtype=scatter.dtype)
                emb_flat += scatter # 이렇게 해줌으로써 cutoff에 따라 범위 안에 못 들고 0으로 채워진 값들을 차근 차근 채워서 (648,128) 모두 채울 수 있는 것!

            embed_shape = shape_list(inp) + [self.d_proj]
            # print('embed_shape',embed_shape) #embed_shape [18, 36, 128]
            embed = tf.reshape(emb_flat, embed_shape) #shape=(18, 36, 128)

        embed *= self.emb_scale #  텐서의 모든 요소에 self.emb_scale을 곱하고 결과를 embed에 다시 저장한다는 의미입니다.

        return embed
    
    
    

@keras_serializable
class TFTransfoXLMainLayer(tf.keras.layers.Layer):
    config_class = TransfoXLConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict

        self.n_token = config.vocab_size

        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.untie_r = config.untie_r

        self.word_emb = TFAdaptiveEmbedding(
            config.vocab_size,
            config.d_embed,
            config.d_model,
            config.cutoffs,
            div_val=config.div_val,
            init_std=config.init_std,
            name="word_emb",
        )

        self.drop = tf.keras.layers.Dropout(config.dropout)

        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.attn_type = config.attn_type

        self.layers = []
        if config.attn_type == 0:  # the default attention
            for i in range(config.n_layer):
                self.layers.append(
                    TFRelPartialLearnableDecoderLayer(
                        config.n_head,
                        config.d_model,
                        config.d_head,
                        config.d_inner,
                        config.dropout,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if self.untie_r else self.r_w_bias,
                        r_r_bias=None if self.untie_r else self.r_r_bias,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                        init_std=config.init_std,
                        output_attentions=self.output_attentions,
                        name=f"layers_._{i}",
                    )
                )
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        if self.attn_type == 0:  # default attention
            self.pos_emb = TFPositionalEmbedding(self.d_model, name="pos_emb")
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

    def build(self, input_shape):
        if not self.untie_r:
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )
        super().build(input_shape)

    def get_input_embeddings(self):
        return self.word_emb

    def set_input_embeddings(self, value):
        raise NotImplementedError

    def backward_compatible(self):
        self.sample_softmax = -1

    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def _prune_heads(self, heads):
        raise NotImplementedError

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer):
                empty = tf.zeros([self.mem_len, bsz, self.d_model])
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        new_mems = []
        end_idx = mlen + tf.math.maximum(0, qlen)
        beg_idx = tf.math.maximum(0, end_idx - tf.convert_to_tensor(self.mem_len))
        for i in range(len(hids)):
            mems[i] = tf.cast(mems[i], dtype=hids[i].dtype)
            cat = tf.concat([mems[i], hids[i]], axis=0)
            tf.stop_gradient(cat)
            new_mems.append(cat[beg_idx:end_idx]) # 항상 메모리의 길이가 400이 되도록 유지

        return new_mems

    def call(
        self,
        input_ids=None,
        mems=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        
        # the original code for Transformer-XL used shapes [len, bsz] but we want a unified interface in the library
        # so we transpose here from shape [bsz, len] to shape [len, bsz]
        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            inputs["input_ids"] = tf.transpose(inputs["input_ids"], perm=(1, 0))

            qlen, bsz = shape_list(inputs["input_ids"])
            # print('inputs["input_ids"]',(qlen, bsz))  #(3:bsz, 36:target len)
        elif inputs["inputs_embeds"] is not None:
            inputs["inputs_embeds"] = tf.transpose(inputs["inputs_embeds"], perm=(1, 0, 2))
            qlen, bsz = shape_list(inputs["inputs_embeds"])[:2]
            # print('inputs["inputs_embeds"]',(qlen, bsz))

        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs["mems"] is None:
            inputs["mems"] = self.init_mems(bsz)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.n_layer

        if inputs["inputs_embeds"] is not None:
            word_emb = inputs["inputs_embeds"]
        else:
            word_emb = self.word_emb(inputs["input_ids"])
        
        # print('확인해보자',inputs["mems"])
        if type(inputs["mems"][0]) is list:
            inputs["mems"] = inputs["mems"][0]
        
        mlen = shape_list(inputs["mems"][0])[0] if inputs["mems"] is not None else 0
        
        # print('qlen',qlen) 
        # print('shape_list',shape_list(inputs["mems"]))
        klen = mlen + qlen

        attn_mask = tf.ones([qlen, qlen])
        mask_u = tf.linalg.band_part(attn_mask, 0, -1)
        mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
        attn_mask_pad = tf.zeros([qlen, mlen])
        dec_attn_mask = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        if self.same_length:
            mask_l = tf.linalg.band_part(attn_mask, -1, 0)
            dec_attn_mask = tf.concat([dec_attn_mask[:, :qlen] + mask_l - mask_dia, dec_attn_mask[:, qlen:]], 1)
        # ::: PyTorch masking code for reference :::
        # if self.same_length:
        #     all_ones = word_emb.new_ones((qlen, klen), dtype=torch.uint8)
        #     mask_len = klen - self.mem_len
        #     if mask_len > 0:
        #         mask_shift_len = qlen - mask_len
        #     else:
        #         mask_shift_len = qlen
        #     dec_attn_mask = (torch.triu(all_ones, 1+mlen)
        #             + torch.tril(all_ones, -mask_shift_len))[:, :, None] # -1
        # else:
        #     dec_attn_mask = torch.triu(
        #         word_emb.new_ones((qlen, klen), dtype=torch.uint8), diagonal=1+mlen)[:,:,None]

        hids = []
        attentions = [] if inputs["output_attentions"] else None

        
        if self.attn_type == 0:  # default
            pos_seq = tf.range(klen - 1, -1, -1.0)

            '''
            pos_seq = tf.Tensor(
            [119. 118. 117. 116. 115. 114. 113. 112. 111. 110. 109. 108. 107. 106. 105.
            104. 103. 102. 101. 100.  99.  98.  97.  96.  95.  94.  93.  92.  91.
            90.  89.  88.  87.  86.  85.  84.  83.  82.  81.  80.  79.  78.  77.
            76.  75.  74.  73.  72.  71.  70.  69.  68.  67.  66.  65.  64.  63.
            62.  61.  60.  59.  58.  57.  56.  55.  54.  53.  52.  51.  50.  49.
            48.  47.  46.  45.  44.  43.  42.  41.  40.  39.  38.  37.  36.  35.
            34.  33.  32.  31.  30.  29.  28.  27.  26.  25.  24.  23.  22.  21.
            20.  19.  18.  17.  16.  15.  14.  13.  12.  11.  10.   9.   8.   7.
            6.   5.   4.   3.   2.   1.   0.], shape=(120,), dtype=float32)
            
            
            '''
            if self.clamp_len > 0:
                pos_seq = tf.minimum(pos_seq, self.clamp_len)
            pos_emb = self.pos_emb(pos_seq) # (pos_seq.len,배치크기, 임베딩 차원)

            core_out = self.drop(word_emb, training=inputs["training"])
            pos_emb = self.drop(pos_emb, training=inputs["training"])

            for i, layer in enumerate(self.layers):
                hids.append(core_out) #레이어의 결과물들이 여기에 저장 된다. 리스트 형태로 
                mems_i = None if inputs["mems"] is None else inputs["mems"][i] # 레이어별로 만들어진 것을 레이어에 맞게 가져옴
                layer_outputs = layer(
                    core_out,
                    pos_emb,
                    dec_attn_mask,
                    mems_i,
                    inputs["head_mask"][i],
                    inputs["output_attentions"],
                    training=inputs["training"],
                )
                core_out = layer_outputs[0] # 리스트에 담겨서 나오니깐 0으로 그냥 값 꺼낸 준거고 (3, 36, 128) 인풋과 같은 사이즈 
                
                
                if inputs["output_attentions"]:
                    attentions.append(layer_outputs[1])
            
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        core_out = self.drop(core_out, training=inputs["training"]) #마지막 레이어의 아웃풋  드랍아웃 후에도 shape=(10, 36, 128)
        
        new_mems = self._update_mems(hids, inputs["mems"], mlen, qlen)  # 각 레이어의 아웃풋과 레이어의 수에 맞게 생성된 메모리가 결합된다. 
 

        # We transpose back here to shape [bsz, len, hidden_dim]
        core_out = tf.transpose(core_out, perm=(1, 0, 2)) # (36,10,128)

        if inputs["output_hidden_states"]:
            # Transpose to library standard shape [bsz, len, hidden_dim] and add last layer
            hids = tuple(tf.transpose(t, perm=(1, 0, 2)) for t in hids)
            hids = hids + (core_out,)
        else:
            hids = None
        if inputs["output_attentions"]:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(tf.transpose(t, perm=(2, 3, 0, 1)) for t in attentions)

        if not inputs["return_dict"]:
            return tuple(v for v in [core_out, new_mems, hids, attentions] if v is not None)

        return TFTransfoXLModelOutput(
            last_hidden_state=core_out,
            mems=new_mems,
            hidden_states=hids,
            attentions=attentions,
        )
        

class TFTransfoXLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TransfoXLConfig
    base_model_prefix = "transformer"

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)
    


@keras_serializable
class TFTransfoXLMLMMainLayer(tf.keras.layers.Layer):
    '''
        기존의 어텐션 마스크 삭제 
    '''
    config_class = TransfoXLConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict

        self.n_token = config.vocab_size

        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.untie_r = config.untie_r

        vocab_size = config.C_vocab_size if config.mode == 'concepts' else config.Q_vocab_size        
        self.word_emb_C  = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=config.d_embed)
        self.word_emb_R  = tf.keras.layers.Embedding(input_dim=config.R_vocab_size, output_dim=config.d_embed)
        self.linear = tf.keras.layers.Dense(self.config.hidden_size)
        self.activation = tf.keras.layers.Activation('gelu')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.config.layer_norm_epsilon)
        self.decoder = tf.keras.layers.Dense(self.config.R_vocab_size, use_bias=False)  # No bias in this layer
    
        

        self.drop = tf.keras.layers.Dropout(config.dropout)

        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.attn_type = config.attn_type

        self.layers = []
        if config.attn_type == 0:  # the default attention
            for i in range(config.n_layer):
                self.layers.append(
                    TFRelPartialLearnableDecoderLayer(
                        config.n_head,
                        config.d_model,
                        config.d_head,
                        config.d_inner,
                        config.dropout,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if self.untie_r else self.r_w_bias,
                        r_r_bias=None if self.untie_r else self.r_r_bias,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                        init_std=config.init_std,
                        output_attentions=self.output_attentions,
                        name=f"layers_._{i}",
                    )
                )
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        if self.attn_type == 0:  # default attention
            self.pos_emb = TFPositionalEmbedding(self.d_model, name="pos_emb")
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

    def build(self, input_shape):
        if not self.untie_r:
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )
        super().build(input_shape)

    def get_input_embeddings(self):
        return self.word_emb

    def set_input_embeddings(self, value):
        raise NotImplementedError

    def backward_compatible(self):
        self.sample_softmax = -1

    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def _prune_heads(self, heads):
        raise NotImplementedError

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer):
                empty = tf.zeros([self.mem_len, bsz, self.d_model])
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        new_mems = []
        end_idx = mlen + tf.math.maximum(0, qlen)
        beg_idx = tf.math.maximum(0, end_idx - tf.convert_to_tensor(self.mem_len))
        for i in range(len(hids)):
            mems[i] = tf.cast(mems[i], dtype=hids[i].dtype)
            cat = tf.concat([mems[i], hids[i]], axis=0)
            tf.stop_gradient(cat)
            new_mems.append(cat[beg_idx:end_idx]) # 항상 메모리의 길이가 400이 되도록 유지

        return new_mems

    def call(
        self,
        concepts=None,
        responses=None,
        mems=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
    
        inputs = input_processing_MLM(
            func=self.call,
            config=self.config,
            concepts=concepts,  
            responses=responses,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        #inputs {'mems': None, 'head_mask': None, 'inputs_embeds': None, 'output_attentions': False, 'output_hidden_states': False, 'return_dict': True, 'training': False, 'input_ids': <tf.Tensor 'dataset_inputs:0' shape=(1, 3) dtype=int32>}

        # the original code for Transformer-XL used shapes [len, bsz] but we want a unified interface in the library
        # so we transpose here from shape [bsz, len] to shape [len, bsz]
        if inputs["concepts"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["concepts"] is not None:
            # print('inputs["input_ids"]',inputs["input_ids"].shape)
            # print('input[]"concepts"]',inputs["concepts"])
            # inputs["concepts"] = tf.transpose(inputs["concepts"], perm=(1, 0))
            # inputs["responses"] = tf.transpose(inputs["responses"], perm=(1, 0))
            # print('inputs["input_ids2222222"]',inputs["input_ids"].shape)

            qlen, bsz = shape_list(inputs["concepts"])
            # print('inputs["input_ids"]',(qlen, bsz))  #(3:bsz, 36:target len)
        elif inputs["inputs_embeds"] is not None:
            inputs["inputs_embeds"] = tf.transpose(inputs["inputs_embeds"], perm=(1, 0, 2))
            qlen, bsz = shape_list(inputs["inputs_embeds"])[:2]
            # print('inputs["inputs_embeds"]',(qlen, bsz))

        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs["mems"] is None:
            inputs["mems"] = self.init_mems(bsz)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.n_layer

        if inputs["inputs_embeds"] is not None:
            word_emb = inputs["inputs_embeds"]
        else:
            word_emb_C = self.word_emb_C(inputs["concepts"])
            word_emb_R = self.word_emb_R(inputs["responses"])
            word_emb = word_emb_C + word_emb_R
        
        # print('확인해보자',inputs["mems"])
        if type(inputs["mems"][0]) is list:
            inputs["mems"] = inputs["mems"][0]
        
        mlen = shape_list(inputs["mems"][0])[0] if inputs["mems"] is not None else 0
  
        klen = mlen + qlen

        # attn_mask = tf.ones([qlen, qlen])
        # mask_u = tf.linalg.band_part(attn_mask, 0, -1)
        # mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
        # attn_mask_pad = tf.zeros([qlen, mlen])
        # dec_attn_mask = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        # if self.same_length:
        #     mask_l = tf.linalg.band_part(attn_mask, -1, 0)
        #     dec_attn_mask = tf.concat([dec_attn_mask[:, :qlen] + mask_l - mask_dia, dec_attn_mask[:, qlen:]], 1)
        # # ::: PyTorch masking code for reference :::
        # # if self.same_length:
        # #     all_ones = word_emb.new_ones((qlen, klen), dtype=torch.uint8)
        #     mask_len = klen - self.mem_len
        #     if mask_len > 0:
        #         mask_shift_len = qlen - mask_len
        #     else:
        #         mask_shift_len = qlen
        #     dec_attn_mask = (torch.triu(all_ones, 1+mlen)
        #             + torch.tril(all_ones, -mask_shift_len))[:, :, None] # -1
        # else:
        #     dec_attn_mask = torch.triu(
        #         word_emb.new_ones((qlen, klen), dtype=torch.uint8), diagonal=1+mlen)[:,:,None]

        hids = []
        attentions = [] if inputs["output_attentions"] else None

        
        if self.attn_type == 0:  # default
            pos_seq = tf.range(klen - 1, -1, -1.0)

            '''
            pos_seq = tf.Tensor(
            [119. 118. 117. 116. 115. 114. 113. 112. 111. 110. 109. 108. 107. 106. 105.
            104. 103. 102. 101. 100.  99.  98.  97.  96.  95.  94.  93.  92.  91.
            90.  89.  88.  87.  86.  85.  84.  83.  82.  81.  80.  79.  78.  77.
            76.  75.  74.  73.  72.  71.  70.  69.  68.  67.  66.  65.  64.  63.
            62.  61.  60.  59.  58.  57.  56.  55.  54.  53.  52.  51.  50.  49.
            48.  47.  46.  45.  44.  43.  42.  41.  40.  39.  38.  37.  36.  35.
            34.  33.  32.  31.  30.  29.  28.  27.  26.  25.  24.  23.  22.  21.
            20.  19.  18.  17.  16.  15.  14.  13.  12.  11.  10.   9.   8.   7.
            6.   5.   4.   3.   2.   1.   0.], shape=(120,), dtype=float32)
            
            
            '''
            if self.clamp_len > 0:
                pos_seq = tf.minimum(pos_seq, self.clamp_len)
            pos_emb = self.pos_emb(pos_seq) # (pos_seq.len,배치크기, 임베딩 차원)

            core_out = self.drop(word_emb, training=inputs["training"])
            pos_emb = self.drop(pos_emb, training=inputs["training"])

            for i, layer in enumerate(self.layers):
                hids.append(core_out) #레이어의 결과물들이 여기에 저장 된다. 리스트 형태로 
                mems_i = None if inputs["mems"] is None else inputs["mems"][i] # 레이어별로 만들어진 것을 레이어에 맞게 가져옴
                layer_outputs = layer(
                    core_out,
                    pos_emb,
                    mems_i,
                    inputs["head_mask"][i],
                    inputs["output_attentions"],
                    training=inputs["training"],
                    dec_attn_mask = None,
                )
                core_out = layer_outputs[0] # 리스트에 담겨서 나오니깐 0으로 그냥 값 꺼낸 준거고 (3, 36, 128) 인풋과 같은 사이즈 
                
                
                if inputs["output_attentions"]:
                    attentions.append(layer_outputs[1])
            
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        core_out = self.drop(core_out, training=inputs["training"]) #마지막 레이어의 아웃풋  드랍아웃 후에도 shape=(10, 36, 128)
        
        new_mems = self._update_mems(hids, inputs["mems"], mlen, qlen)  # 각 레이어의 아웃풋과 레이어의 수에 맞게 생성된 메모리가 결합된다. 

        x = self.linear(core_out)
        x = self.activation(x) # gelu
        x = self.layer_norm(x)
        core_out = self.decoder(x) #output = responses size

        # We transpose back here to shape [bsz, len, hidden_dim]
        # core_out = tf.transpose(core_out, perm=(1, 0, 2)) # (36,10,128)

       




        if inputs["output_hidden_states"]:
            # Transpose to library standard shape [bsz, len, hidden_dim] and add last layer
            hids = tuple(tf.transpose(t, perm=(1, 0, 2)) for t in hids)
            hids = hids + (core_out,)
        else:
            hids = None
        if inputs["output_attentions"]:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(tf.transpose(t, perm=(2, 3, 0, 1)) for t in attentions)

        if not inputs["return_dict"]:
            return tuple(v for v in [core_out, new_mems, hids, attentions] if v is not None)

        return TFTransfoXLModelOutput(
            last_hidden_state=core_out,
            mems=new_mems,
            hidden_states=hids,
            attentions=attentions,
        )
        

class TFTransfoXLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TransfoXLConfig
    base_model_prefix = "transformer"

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)
    
    
    

@dataclass
class TFTransfoXLModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: tf.Tensor = None
    mems: List[tf.Tensor] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFTransfoXLLMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        losses (`tf.Tensor` of shape *(batch_size, sequence_length-1)*, *optional*, returned when `labels` is provided)
            Language modeling losses (not reduced).
        prediction_scores (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    logit: Optional[tf.Tensor] = None
    prediction_scores: tf.Tensor = None
    mems: List[tf.Tensor] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    labels: Optional[tf.Tensor] = None



@dataclass
class TFTransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    mems: List[tf.Tensor] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    
    

TRANSFO_XL_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all
    the tensors in the first argument of the model call function: `model(inputs)`.

    If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
    the first positional argument :

    - a single Tensor with `input_ids` only and nothing else: `model(inputs_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    </Tip>

    Parameters:
        config ([`TransfoXLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model
            weights.
"""

TRANSFO_XL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.__call__`] and [`PreTrainedTokenizer.encode`] for
            details.

            [What are input IDs?](../glossary#input-ids)
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems
            given to this model should not be passed as `input_ids` as they have already been computed.
        head_mask (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""




@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs): # *inputs 란 config 다음 자리에 여러개의 인수가 올 수 있음을 의미 
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )

    def call(
        self,
        input_ids=None,
        mems=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        #input_ids = shape =(70,60) 
        #mems = shape =(70,60) 
        outputs = self.transformer(
            input_ids=inputs["input_ids"],
            mems=inputs["mems"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        return outputs

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFTransfoXLModelOutput(
            last_hidden_state=output.last_hidden_state,
            mems=tf.convert_to_tensor(output.mems),
            hidden_states=hs,
            attentions=attns,
        )
        
        

@add_start_docstrings(
    """
    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLLMHeadModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")
        self.sample_softmax = config.sample_softmax
        assert (
            self.sample_softmax <= 0
        ), "Sampling from the softmax is not implemented yet. Please look at issue: #3310: https://github.com/huggingface/transformers/issues/3310"

        self.crit = TFAdaptiveSoftmaxMask(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, name="crit"
        )

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError()

    def get_output_embeddings(self):
        """Double-check if you are using adaptive softmax."""
        if len(self.crit.out_layers) > 0:
            return self.crit.out_layers[-1]
        return None

    def reset_memory_length(self, mem_len):
        self.transformer.reset_memory_length(mem_len)

    def init_mems(self, bsz):
        return self.transformer.init_mems(bsz)

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        mems=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
    
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None:
            bsz, tgt_len = shape_list(inputs["input_ids"])[:2]
        else:
            bsz, tgt_len = shape_list(inputs["inputs_embeds"])[:2]

        transformer_outputs = self.transformer(
            inputs["input_ids"],
            inputs["mems"],
            inputs["head_mask"],
            inputs["inputs_embeds"],
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            inputs["return_dict"],
            training=inputs["training"],
        )

        last_hidden = transformer_outputs[0] # core output (36,10,128)
        # print('tgt_len',tgt_len) # 10
        pred_hid = last_hidden[:, -tgt_len:]

        softmax_output = self.crit(pred_hid, labels, training=inputs["training"])
        prediction_scores = softmax_output if labels is None else ()
        
        # loss = tf.reshape(softmax_output, [bsz, tgt_len, -1]) if labels is not None else None
        
        # loss = tf.reduce_mean(loss)
        loss = self.crit.losses
        
        if not inputs["return_dict"]:
            return (softmax_output,) + transformer_outputs[1:]

        return TFTransfoXLLMHeadModelOutput(
            loss = loss,
            prediction_scores=prediction_scores,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFTransfoXLLMHeadModelOutput(
            prediction_scores=output.prediction_scores,
            mems=tf.convert_to_tensor(output.mems),
            hidden_states=hs,
            attentions=attns,
        )

    def prepare_inputs_for_generation(self, inputs, past, **model_kwargs):
        inputs = {"input_ids": inputs}

        # if past is defined in model kwargs then use it for faster decoding
        if past:
            inputs["mems"] = past

        return inputs


@add_start_docstrings(
    """
    The Transformer-XL Model with a Masked language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLMLMHeadModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TFTransfoXLMLMMainLayer(config, name="transformer")
        self.sample_softmax = config.sample_softmax
        assert (
            self.sample_softmax <= 0
        ), "Sampling from the softmax is not implemented yet. Please look at issue: #3310: https://github.com/huggingface/transformers/issues/3310"

        # self.crit = TFAdaptiveSoftmaxMask(
        #     config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, name="crit"
        # )
        self.crit = keras.losses.SparseCategoricalCrossentropy(
           
            )   
        # mlm head 부분 추가
      

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError()

    def get_output_embeddings(self):
        """Double-check if you are using adaptive softmax."""
        if len(self.crit.out_layers) > 0:
            return self.crit.out_layers[-1]
        return None

    def reset_memory_length(self, mem_len):
        self.transformer.reset_memory_length(mem_len)

    def init_mems(self, bsz):
        return self.transformer.init_mems(bsz)

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        concepts=None,
        responses=None,
        mems=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
    
        inputs = input_processing_MLM(
            func=self.call,
            config=self.config,
            concepts=concepts,  
            responses=responses,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        if inputs["concepts"] is not None:
            bsz, tgt_len = shape_list(inputs["concepts"])[:2]
        else:
            bsz, tgt_len = shape_list(inputs["inputs_embeds"])[:2]

        transformer_outputs = self.transformer(
            inputs["concepts"],
            inputs["responses"],
            inputs["mems"],
            inputs["head_mask"],
            inputs["inputs_embeds"],
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            inputs["return_dict"],
            training=inputs["training"],

        )

        last_hidden = transformer_outputs[0] # core output (36,10,128)
        # print('last_hidden',last_hidden[0])
        # mlm head
        

        '''
        loss_mx = labels != -100
        output = output[loss_mx].view(-1, self.tokenizer.vocab_size)
        labels = labels[loss_mx].view(-1)
        mask한 부분의 값만 계산할 수 있도록 

        # '''
        # output=tf.reshape(output,[-1,self.config.R_vocab_size])
        # labels=tf.reshape(labels,[-1])
        # # loss_mx = labels != -100
        # # output = tf.reshape(output[loss_mx],[-1,self.config.R_vocab_size])
        # # labels = tf.reshape(labels[loss_mx],[-1])
        
        # print('output',output)
        # tf.print('labels',labels)   

        # loss = self.crit(output, labels)

        # print('tgt_len',tgt_len) # 10
        # pred_hid = last_hidden[:, -tgt_len:]

        # softmax_output = self.crit(pred_hid, labels, training=inputs["training"])
        # prediction_scores = softmax_output if labels is None else ()
        
        # loss = tf.reshape(softmax_output, [bsz, tgt_len, -1]) if labels is not None else None
        
        # loss = tf.reduce_mean(loss)
        # loss = self.crit.losses
        
        # if not inputs["return_dict"]:
        #     return (softmax_output,) + transformer_outputs[1:]

        return TFTransfoXLLMHeadModelOutput(
            logit = last_hidden,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            prediction_scores=None,#prediction_scores,
            labels =labels

        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFTransfoXLLMHeadModelOutput(
            prediction_scores=output.prediction_scores,
            mems=tf.convert_to_tensor(output.mems),
            hidden_states=hs,
            attentions=attns,
        )

    

@add_start_docstrings(
    """
    The Transfo XL Model transformer with a sequence classification head on top (linear layer).

    [`TFTransfoXLForSequenceClassification`] uses the last token in order to do the classification,
    as other causal models (e.g. GPT-1,GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take
    the last value in each row of the batch).
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLForSequenceClassification(TFTransfoXLPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.score = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.init_range),
            name="score",
            use_bias=False,
        )
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")

    def get_output_embeddings(self):
        return self.transformer.word_emb

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLSequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        mems=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ..., config.vocab_size - 1]`.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        transformer_outputs = self.transformer(
            input_ids=inputs["input_ids"],
            mems=inputs["mems"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        in_logits = None
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if inputs["input_ids"] is not None:
                sequence_lengths = (
                    tf.reduce_sum(
                        tf.cast(
                            tf.math.not_equal(inputs["input_ids"], self.config.pad_token_id),
                            dtype=inputs["input_ids"].dtype,
                        ),
                        -1,
                        keepdims=False,
                    )
                    - 1
                )
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        loss = None

        if inputs["labels"] is not None:
            if input_ids is not None:
                batch_size, sequence_length = shape_list(inputs["input_ids"])[:2]
            else:
                batch_size, sequence_length = shape_list(inputs["inputs_embeds"])[:2]
            assert (
                self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."

            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]

            loss = self.compute_loss(
                tf.reshape(inputs["labels"], [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels])
            )

        pooled_logits = in_logits if in_logits is not None else logits

        if not inputs["return_dict"]:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFTransfoXLSequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        ), self.config.num_labels

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFTransfoXLSequenceClassifierOutputWithPast(
            logits=output.logits, mems=tf.convert_to_tensor(output.mems), hidden_states=hs, attentions=attns
        )
