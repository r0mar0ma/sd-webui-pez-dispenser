import time
import copy
import open_clip
import torch

try:
    from transformers.optimization import Adafactor
except:
    pass

from modules import shared

TORCH_COMPILE_LEVEL_OFF = 0
TORCH_COMPILE_LEVEL_MAX = 3

optimizers = {
    "AdamW": "Default",
    "Lion": "Generally faster",
    "SGD": "Simpler calculations per step, slightly higher lr",
    "Adafactor": "Less memory, faster for large models"
}

class State:
    installed = False

state = State()

##################################################
#try:
#    from sentence_transformers.util import (semantic_search, dot_score, normalize_embeddings)
#    state.installed = True
#except:
#    print("pez-dispenser error: No sentence_transformers package installed")
##################################################

import torch
from torch import Tensor, device
from typing import List, Callable
import numpy as np

state.installed = True

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def semantic_search(query_embeddings: Tensor,
                    corpus_embeddings: Tensor,
                    query_chunk_size: int = 100,
                    corpus_chunk_size: int = 500000,
                    top_k: int = 10,
                    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)


    #Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx+query_chunk_size], corpus_embeddings[corpus_start_idx:corpus_start_idx+corpus_chunk_size])

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    #Sort and strip to top_k results
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    return queries_result_list

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

def normalize_embeddings(embeddings: Tensor):
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

##################################################

# Simple Lion optimizer implementation
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Perform weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update moving average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update weights
                update = exp_avg.sign()
                p.add_(update, alpha=-group['lr'])

##################################################

def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)

        #if print_hits:
        #    all_hits = []
        #    for hit in hits:
        #        all_hits.append(hit[0]["score"])
        #    print(f"mean hits:{mean(all_hits)}")
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz,seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices

def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append("|".join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts

def get_target_feature_images(model, preprocess, device, target_images):
    with torch.no_grad():
        curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
        curr_images = torch.concatenate(curr_images).to(device)
        cast_dtype = model.transformer.get_cast_dtype()
        all_target_features = model.encode_image(curr_images.to(cast_dtype))

    return all_target_features


def get_target_feature_prompts(model, tokenizer_funct, device, target_prompts):
    texts = tokenizer_funct(target_prompts).to(device)
    all_target_features = model.encode_text(texts)

    return all_target_features


def initialize_prompt(tokenizer, token_embedding, device, prompt_len, prompt_bs):

    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(len(tokenizer.encoder), (prompt_bs, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * prompt_bs).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False
    
    return prompt_embeds, dummy_embeds, dummy_ids


#def encode_text_embedding(model, text_embedding, ids, avg_text=False):
#    cast_dtype = model.transformer.get_cast_dtype()
#
#    x = text_embedding + model.positional_embedding.to(cast_dtype)
#    x = x.permute(1, 0, 2)  # NLD -> LND
#    x = model.transformer(x, attn_mask=model.attn_mask)
#    x = x.permute(1, 0, 2)  # LND -> NLD
#    x = model.ln_final(x)
#
#    # x.shape = [batch_size, n_ctx, transformer.width]
#    # take features from the eot embedding (eot_token is the highest number in each sequence)
#    if avg_text:
#        x = x[torch.arange(x.shape[0]), :ids.argmax(dim=-1)]
#        x[:, 1:-1]
#        x = x.mean(dim=1) @ model.text_projection
#    else:
#        x = x[torch.arange(x.shape[0]), ids.argmax(dim=-1)] @ model.text_projection
#
#    return x
#
#def forward_text_embedding(model, embeddings, ids, image_features, avg_text=False, return_feature=False):
#    text_features = encode_text_embedding(model, embeddings, ids, avg_text=avg_text)
#    if return_feature:
#        return text_features
#
#    # normalized features
#    image_features = image_features / image_features.norm(dim=1, keepdim=True)
#    text_features = text_features / text_features.norm(dim=1, keepdim=True)
#
#    # cosine similarity as logits
#    # logit_scale = model.logit_scale.exp()
#    logits_per_image = image_features @ text_features.t()
#    logits_per_text = logits_per_image.t()
#
#    # shape = [global_batch_size, global_batch_size]
#    return logits_per_image, logits_per_text
    
def _forward_text_embedding(model, embeddings, ids, image_features):
    cast_dtype = model.transformer.get_cast_dtype()

    x = embeddings.to(cast_dtype) + model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x)

    # Optimized for CUDA graphs
    batch_size, seq_len, embed_dim = x.shape
    max_indices = ids.argmax(dim=-1)
    gather_indices = max_indices.reshape(batch_size, 1, 1).expand(-1, -1, embed_dim)
    extracted_features = torch.gather(x, dim=1, index=gather_indices).squeeze(1)
    e = extracted_features @ model.text_projection

    """
    # Original code
    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), ids.argmax(dim=-1)] @ model.text_projection

    assert torch.allclose(x, e)

    text_features = x
    """
    text_features = e

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    # logit_scale = model.logit_scale.exp()
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text

_forward_text_embedding_compiled = None
_forward_text_embedding_compiled_level = 0
_forward_text_embedding_compiled_valid = False
_forward_text_embedding_compiled_cuda = False

def forward_text_embedding(model, embeddings, ids, image_features):
    global _forward_text_embedding_compiled_valid
    if _forward_text_embedding_compiled_valid:
        try:
            if _forward_text_embedding_compiled_cuda:
                torch.compiler.cudagraph_mark_step_begin()
            return _forward_text_embedding_compiled(model, embeddings, ids, image_features)
        except Exception as e:
            print(f"Unable to use compiled forward_text_embedding(), reverting to plain one: {e}")
            _forward_text_embedding_compiled_valid = False
    return _forward_text_embedding(model, embeddings, ids, image_features)

def get_torch_compile_mode(torch_compile_level):
    if torch_compile_level <= 1:
        return "default"
    elif torch_compile_level == 2:
        return "max-autotune-no-cudagraphs"
    else:
        return "max-autotune"

def reset_forward_text_embedding_compiled():
    global _forward_text_embedding_compiled
    global _forward_text_embedding_compiled_level
    global _forward_text_embedding_compiled_valid
    global _forward_text_embedding_compiled_cuda
    
    _forward_text_embedding_compiled = None
    _forward_text_embedding_compiled_level = 0
    _forward_text_embedding_compiled_valid = False
    _forward_text_embedding_compiled_cuda = False
    
    try:
        torch._dynamo.reset()
    except:
        pass

def init_forward_text_embedding(torch_compile_level=0):
    global _forward_text_embedding_compiled
    global _forward_text_embedding_compiled_level
    global _forward_text_embedding_compiled_valid
    global _forward_text_embedding_compiled_cuda

    if torch_compile_level <= 0:
        reset_forward_text_embedding_compiled()
    else:
        if _forward_text_embedding_compiled_level != torch_compile_level:
            reset_forward_text_embedding_compiled()
            _forward_text_embedding_compiled = torch.compile(_forward_text_embedding, mode = get_torch_compile_mode(torch_compile_level))
            _forward_text_embedding_compiled_level = torch_compile_level

        _forward_text_embedding_compiled_valid = True

        if torch_compile_level == 1:
            _forward_text_embedding_compiled_cuda = False
        elif torch_compile_level == 2:
            _forward_text_embedding_compiled_cuda = False
        else:
            _forward_text_embedding_compiled_cuda = True



def optimize_prompt_loop(
    model, 
    tokenizer, 
    token_embedding, 
    all_target_features, 
    device, prompt_len, 
    opt_iters, 
    lr, 
    weight_decay, 
    prompt_bs, 
    print_step, 
    batch_size, 
    optimizer,
    on_progress, 
    progress_steps, 
    progress_args
):
    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, device, prompt_len, prompt_bs)
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    if optimizer == "AdamW":
        input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)
    elif optimizer == "Lion":
        input_optimizer = Lion([prompt_embeds], lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    elif optimizer == "SGD":
        input_optimizer = torch.optim.SGD([prompt_embeds], lr=lr, momentum=0.9)
    elif optimizer == "Adafactor":
        input_optimizer = Adafactor([prompt_embeds], lr=lr, relative_step=False)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    best_sim = 0
    best_text = ""
    
    for step in range(opt_iters):
        if shared.state.interrupted:
            break
        if not on_progress is None:
            for progress_step in progress_steps:
                if step % progress_step == 0:
                    on_progress(step, opt_iters, best_text, progress_args)
                    break
    
        # randomly sample sample images and get features
        if batch_size is None:
            target_features = all_target_features
        else:
            curr_indx = torch.randperm(len(all_target_features))
            target_features = all_target_features[curr_indx][0:batch_size]
            
        universal_target_features = all_target_features

        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding, print_hits=False)

        # get cosine similarity score with all target features
        with torch.no_grad():
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = projected_embeds.reshape(-1, p_dim)
            logits_per_image, _ = forward_text_embedding(model, padded_embeds, dummy_ids, universal_target_features)
            scores_per_prompt = logits_per_image.mean(dim=0)
            universal_cosim_score = scores_per_prompt.max().item()
            best_indx = scores_per_prompt.argmax().item()
        
        tmp_embeds = prompt_embeds.detach().clone()
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True
        
        # padding
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)
            
        logits_per_image, _ = forward_text_embedding(model, padded_embeds, dummy_ids, target_features)
        cosim_scores = logits_per_image
        loss = 1 - cosim_scores.mean()
        
        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
        
        input_optimizer.step()
        input_optimizer.zero_grad()

        curr_lr = input_optimizer.param_groups[0]["lr"]
        cosim_scores = cosim_scores.mean().item()

        decoded_text = decode_ids(nn_indices, tokenizer)[best_indx]

        if print_step is not None and (step % print_step == 0 or step == opt_iters-1):
            print(f"step: {step}, lr: {curr_lr}, cosim: {universal_cosim_score:.3f}, text: {decoded_text}")
        
        if best_sim < universal_cosim_score:
            best_sim = universal_cosim_score
            
            best_text = decoded_text

    if not on_progress is None:
        on_progress(step + 1, opt_iters, best_text, progress_args)

    if print_step is not None:
        print()
        print(f"best cosine sim: {best_sim}")
        print(f"best prompt: {best_text}")

    return best_text


def optimize_prompt(
    model,
    preprocess,
    device,
    clip_model,
    prompt_len,
    opt_iters,
    lr,
    weight_decay,
    prompt_bs,
    print_step,
    batch_size,
    target_images=None,
    target_prompts=None,
    on_progress=None,
    progress_steps=[1],
    progress_args=None,
    optimizer = "AdamW",
    torch_compile_level=0
):
    if not state.installed:
        raise ModuleNotFoundError("Some required packages are not installed")

    init_forward_text_embedding(torch_compile_level)

    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer

    # get target features
    image_target_features = None
    text_target_features = None
    if (not target_images is None) and (len(target_images) > 0):
        image_target_features = get_target_feature_images(model, preprocess, device, target_images)
    if (not target_prompts is None) and (len(target_prompts) > 0):
        tokenizer_funct = open_clip.get_tokenizer(clip_model)
        text_target_features = get_target_feature_prompts(model, tokenizer_funct, device, target_prompts)

    if (not image_target_features is None) and (not text_target_features is None):
        all_target_features = torch.cat((text_target_features, image_target_features), dim = 0)
    elif not image_target_features is None:
        all_target_features = image_target_features
    elif not text_target_features is None:
        all_target_features = text_target_features
    else:
        raise ValueError("No input images or prompts")
    
    # optimize prompt
    learned_prompt = optimize_prompt_loop(
        model, tokenizer, token_embedding, all_target_features, device,
        prompt_len, opt_iters, lr, weight_decay, prompt_bs, print_step, batch_size,
        optimizer,
        on_progress, progress_steps, progress_args
    )
    
    return learned_prompt
