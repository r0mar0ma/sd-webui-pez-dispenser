import copy
import os
import json
import time
import argparse
import traceback
import torch
import open_clip
import gradio as gr
#import common.optim_utils as utils
from modules import devices, scripts, script_callbacks, ui, shared, progress, extra_networks
from modules.processing import process_images, Processed
from modules.ui_components import ToolButton

VERSION = "1.2.1"

#################################
########## optim_utils ##########
#################################

#import time
#import copy
#import open_clip
#import torch
#from modules import shared

class State:
    installed = False

state = State()

try:
    from sentence_transformers.util import (semantic_search, dot_score, normalize_embeddings)
    state.installed = True
except:
    print("pez-dispenser error: No sentence_transformers package installed")


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

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")
        
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
        all_target_features = model.encode_image(curr_images)

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


def encode_text_embedding(model, text_embedding, ids, avg_text=False):
    cast_dtype = model.transformer.get_cast_dtype()

    x = text_embedding + model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    if avg_text:
        x = x[torch.arange(x.shape[0]), :ids.argmax(dim=-1)]
        x[:, 1:-1]
        x = x.mean(dim=1) @ model.text_projection
    else:
        x = x[torch.arange(x.shape[0]), ids.argmax(dim=-1)] @ model.text_projection

    return x
    
def forward_text_embedding(model, embeddings, ids, image_features, avg_text=False, return_feature=False):
    text_features = encode_text_embedding(model, embeddings, ids, avg_text=avg_text)

    if return_feature:
        return text_features

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    # logit_scale = model.logit_scale.exp()
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


def optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, device, prompt_len, opt_iters, lr, weight_decay, prompt_bs, print_step, batch_size, on_progress, progress_steps, progress_args):
    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, device, prompt_len, prompt_bs)
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

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


def optimize_prompt(model, preprocess, device, clip_model, prompt_len, opt_iters, lr, weight_decay, prompt_bs, print_step, batch_size, target_images=None, target_prompts=None, on_progress=None, progress_steps=[1], progress_args=None):
    if not state.installed:
        raise ModuleNotFoundError("Some required packages are not installed")

    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer

    # get target features
    if not target_images is None:
        all_target_features = get_target_feature_images(model, preprocess, device, target_images)
    elif not target_prompts is None:
        tokenizer_funct = open_clip.get_tokenizer(clip_model)
        all_target_features = get_target_feature_prompts(model, tokenizer_funct, device, target_prompts)
    else:
        raise ValueError("No input images or prompts")
    
    # optimize prompt
    learned_prompt = optimize_prompt_loop(
        model, tokenizer, token_embedding, all_target_features, device,
        prompt_len, opt_iters, lr, weight_decay, prompt_bs, print_step, batch_size,
        on_progress, progress_steps, progress_args
    )
    
    return learned_prompt

################################
########## optim_utils ##########
#################################

ALLOW_DEVICE_SELECTION = False
INPUT_IMAGES_COUNT = 5

unload_symbol = '\u274c' # delete
#unload_symbol = '\u267b' # recycle

VERSION_HTML = f'Version <a href="https://github.com/r0mar0ma/sd-webui-pez-dispenser/blob/main/CHANGELOG.md" target="_blank">{VERSION}</a>'

class ThisState:

    def __init__(self):
        self.reset()

    def start_progress(self, title):
        self.progress_title = title
        self.progress_started = time.time()

    def reset(self):
        self.model_index = 0
        self.model_device_name = devices.get_optimal_device_name()
        self.model = None
        self.preprocess = None
        self.clip_model = None
        self.progress_title = ""
        self.progress_started = time.time()

this = ThisState()

########## Arguments ##########

config_dir = scripts.basedir()
config_file_path = os.path.join(config_dir, "config.json")

args = argparse.Namespace()
# default args from hard-prompts-made-easy
args.__dict__.update(json.loads("""
{
    "prompt_len": 8,
    "iter": 1000,
    "lr": 0.1,
    "weight_decay": 0.1,
    "prompt_bs": 1,
    "loss_weight": 1.0,
    "print_step": null,
    "batch_size": 1,
    "clip_model": "ViT-L-14",
    "clip_pretrain": "openai",
    "device": null
}
"""))
if os.path.isfile(config_file_path):
    with open(config_file_path, encoding='utf-8') as f:
        args.__dict__.update(json.load(f))

def show_tab():
    try:
        return shared.opts.pezdispenser_ui_mode in [ "Tab and Script", "Tab only" ]
    except:
        return True

def show_script():
    try:
        return shared.opts.pezdispenser_ui_mode in [ "Tab and Script", "Script only" ]
    except:
        return True

########## Devices ##########

def get_device_display_name(device_name):
    if device_name == "cpu":
        return "CPU"
    if device_name.startswith("cuda"):
        props = torch.cuda.get_device_properties(torch.device(device_name))
        prefix = "GPU"
        if device_name.startswith("cuda:"):
            prefix += device_name[5:]
        return f"{prefix}: {props.name} ({round(props.total_memory / (1024 * 1024 * 1024))}GB)"
    return device_name

available_devices = []

def append_available_device(device_name, prefix = ""):
    available_devices.append((device_name, prefix + get_device_display_name(device_name)))

if ALLOW_DEVICE_SELECTION:
    append_available_device(devices.get_optimal_device_name(), prefix = "(Default) ")

    if torch.cuda.is_available():
        append_available_device("cpu")
        for i in range(torch.cuda.device_count()):
            append_available_device(f"cuda:{i}")

    if not args.device is None and args.device in [n for n, _ in available_devices]:
        this.model_device_name = args.device

########## Models ##########

pretrained_models = [
    ("SD 1.5 (ViT-L-14, openai)", "ViT-L-14", "openai"),
    ("SD 2.0, Midjourney (ViT-H-14, laion2b_s32b_b79k)", "ViT-H-14", "laion2b_s32b_b79k")
]
for m, p in open_clip.pretrained.list_pretrained(as_str = False):
    pretrained_models.append((f"{m}, {p}", m, p))

for i in range(len(pretrained_models)):
    if pretrained_models[i][1] == args.clip_model and pretrained_models[i][2] == args.clip_pretrain:
        this.model_index = i
        break

def unload_model():
    if this.model is None and this.preprocess is None:
        return "Model was not loaded"

    is_cpu = this.model_device_name == "cpu"
    device = torch.device(this.model_device_name)

    if not is_cpu:
        memory_used_pre = torch.cuda.memory_allocated(device)

    if not this.model is None:
        del this.model
        this.model = None
    if not this.preprocess is None:
        del this.preprocess
        this.preprocess = None
    
    if is_cpu:
        msg = "Model unloaded"
    else:
        memory_freed = memory_used_pre - torch.cuda.memory_allocated(device)
        msg = f"Model unloaded, GPU memory freed: {(memory_freed / 1048576):.2f} MB"

    print(msg)
    return msg



def load_model(index, device_name):
    if this.model is None or this.preprocess is None or this.model_index != index or this.model_device_name != device_name:
        _, clip_model, clip_pretrain = pretrained_models[index];

        unload_model()

        print(f"Loading model: {clip_model}:{clip_pretrain}, device: {get_device_display_name(device_name)}")

        is_cpu = device_name == "cpu"
        device = torch.device(device_name)
        if not is_cpu:
            memory_used_pre = torch.cuda.memory_allocated(device)

        model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained = clip_pretrain, device = device)

        this.model = model
        this.preprocess = preprocess
        this.clip_model = clip_model
        this.model_index = index
        this.model_device_name = device_name

        if is_cpu:
            print("Model loaded")
        else:
            memory_taken = torch.cuda.memory_allocated(device) - memory_used_pre
            print(f"Model loaded, GPU memory taken: {(memory_taken / 1048576):.2f} MB")

    return this.model, this.preprocess, this.clip_model

########## Processing ##########

def parse_prompt(prompt):
    if prompt is None:
        return None, ""

    parsed_extra_networks = ""
    parsed_prompt, extra_network_data = extra_networks.parse_prompt(prompt)
    for extra_network_name, extra_network_args in extra_network_data.items():
        for arg in extra_network_args:
            parsed_extra_networks += f"<{extra_network_name}:{':'.join(arg.items)}>"
    
    parsed_prompts = list(filter(lambda s: len(s) > 0 , [ p.strip() for p in parsed_prompt.split("BREAK") ]))
    if len(parsed_prompts) == 0:
        parsed_prompts = None
    
    return parsed_prompts, parsed_extra_networks

def on_progress(step, total, prompt, progress_args):
    if step == 0:
        this.progress_started = time.time()

    progress = step * 100 // total

    processing_time = time.time() - this.progress_started
    speed = (step / processing_time) if processing_time > 0 else 0

    print(f"\r{this.progress_title}: {progress}% ({speed:.2f}it/s)", end = "", flush = True)

    shared.state.job_no = step - 1
    shared.state.nextjob()

def inference(task_id, model_index, device_name_index, prompt_length, iterations_count, lr, weight_decay, prompt_bs, batch_size, target_images = None, target_prompt = None):
    progress.add_task_to_queue(task_id)
    shared.state.begin()
    progress.start_task(task_id)
    shared.state.textinfo = "Preparing..."
    shared.state.job_count = iterations_count

    res = "", ""

    try:
        if not state.installed:
            raise ModuleNotFoundError("Some required packages are not installed. Please restart WebUI to install them automatically.")

        parsed_prompts, parsed_extra_networks = parse_prompt(target_prompt)
        parsed_images = list() if target_images is None else list(filter(lambda i: not i is None, target_images))

        if len(parsed_images) == 0 and parsed_prompts is None:
            raise ValueError("Nothing to process")

        device_name = available_devices[device_name_index][0] if ALLOW_DEVICE_SELECTION else this.model_device_name

        shared.state.textinfo = "Loading model..."
        model, preprocess, clip_model = load_model(model_index, device_name)

        prompt_len = min(int(prompt_length), 75) if prompt_length is not None else 8
        iter = int(iterations_count) if iterations_count is not None else 1000

        shared.state.textinfo = "Processing..."

        prompt = optimize_prompt(
            model,
            preprocess,
            torch.device(device_name),
            clip_model,
            prompt_len,
            iter,
            float(lr),
            float(weight_decay),
            int(prompt_bs),
            None,
            int(batch_size),
            target_images = parsed_images if len(parsed_images) > 0 else None,
            target_prompts = parsed_prompts if not parsed_prompts is None else None,
            on_progress = on_progress,
            progress_steps = [ max(iter // 100, 10) ]
        )

        print("")
        processing_time = time.time() - shared.state.time_start
        res = prompt + parsed_extra_networks, f"Time taken: {processing_time:.2f} sec"

    except Exception as ex:
        print("")
        traceback.print_exception(ex)
        res = "", f"{ex.__class__.__name__}: {ex}"

    progress.finish_task(task_id)
    shared.state.end()
    shared.state.skipped = False
    shared.state.interrupted = False
    shared.state.job_count = 0
    shared.state.textinfo = ""

    return res

def inference_image(task_id, model_index, device_name_index, prompt_length, iterations_count, lr, weight_decay, prompt_bs, batch_size, *target_images):
    this.start_progress("Processing image")
    return inference(task_id, model_index, device_name_index, prompt_length, iterations_count, lr, weight_decay, prompt_bs, batch_size, target_images = target_images)

def inference_text(task_id, model_index, device_name_index, prompt_length, iterations_count, lr, weight_decay, prompt_bs, batch_size, target_prompt):
    this.start_progress("Processing prompt")
    return inference(task_id, model_index, device_name_index, prompt_length, iterations_count, lr, weight_decay, prompt_bs, batch_size, target_prompt = target_prompt if not target_prompt is None and target_prompt != "" else None)

def interrupt():
    shared.state.interrupt()

########## Tab ##########

def wrap_gradio_gpu_call(func):
    def f(*args, **kwargs):

        # if the first argument is a string that says "task(...)", it is treated as a job id
        if len(args) > 0 and type(args[0]) == str and args[0][0:5] == "task(" and args[0][-1] == ")":
            id_task = args[0]
            progress.add_task_to_queue(id_task)
        else:
            id_task = None

        with queue_lock:
            shared.state.begin()
            progress.start_task(id_task)

            try:
                res = func(*args, **kwargs)
            finally:
                progress.finish_task(id_task)

            shared.state.end()

        return res

    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)

def find_prompt(fields):
    field = [x for x in fields if x[1] == "Prompt"][0][0]
    return field

def create_tab():

    input_images = list()

    with gr.Blocks(analytics_enabled = False) as tab:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Tab("Image to prompt"):
                        for i in range(1, INPUT_IMAGES_COUNT + 1):
                            with gr.Tab(f"Image {i}"):
                                input_images.append(gr.Image(type = "pil", label = "Target image", show_label = False, elem_id = f"pezdispenser_input_image_{i}"))
                        process_image_button = gr.Button("Generate prompt", variant = "primary", elem_id = "pezdispenser_process_image_button")
                        setattr(process_image_button, "do_not_save_to_config", True)

                    with gr.Tab("Long prompt to short prompt"):
                        input_text = gr.TextArea(label = "Target prompt", show_label = False, interactive = True, elem_id = "pezdispenser_input_text")
                        process_text_button = gr.Button("Distill prompt", variant = "primary", elem_id = "pezdispenser_process_text_button")
                        setattr(process_text_button, "do_not_save_to_config", True)

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            opt_model = gr.Dropdown(label = "Model", choices = [n for n, _, _ in pretrained_models], type = "index",
                                value = pretrained_models[this.model_index][0], elem_id = "pezdispenser_opt_model")
                            unload_model_button = ToolButton(unload_symbol, elem_id = "pezdispenser_unload_model_button")
                            setattr(unload_model_button, "do_not_save_to_config", True)

                    if ALLOW_DEVICE_SELECTION:
                        with gr.Column():
                            opt_device = gr.Dropdown(
                                label = "Process on",
                                choices = [d for _, d in available_devices] if ALLOW_DEVICE_SELECTION else ["Default"],
                                type = "index",
                                value = next((d for n, d in available_devices if n == this.model_device_name), available_devices[0][1]) if ALLOW_DEVICE_SELECTION else ["Default"],
                                elem_id = "pezdispenser_opt_device",
                                visible = ALLOW_DEVICE_SELECTION
                            )
                    else:
                        opt_device = gr.Dropdown(
                            choices = ["Default"],
                            type = "index",
                            value = ["Default"],
                            elem_id = "pezdispenser_opt_device",
                            visible = False
                        )

                with gr.Row():
                    with gr.Column():
                        opt_prompt_length = gr.Slider(label = "Prompt length (optimal 8-16)", minimum = 1, maximum = 75, step = 1, value = args.prompt_len, elem_id = "pezdispenser_opt_prompt_length")
                    with gr.Column():
                        opt_num_step = gr.Slider(label = "Optimization steps (optimal 1000-3000)", minimum = 1, maximum = 10000, step = 1, value = args.iter, elem_id = "pezdispenser_opt_num_step")

                with gr.Row():
                    with gr.Accordion("Advanced", open = False):
                        with gr.Row():
                            with gr.Column():
                                opt_lr = gr.Textbox(label = "Learning rate for AdamW optimizer (default 0.1)", value = args.lr, lines = 1, max_lines = 1, elem_id = "pezdispenser_opt_lr")
                                opt_weight_decay = gr.Textbox(label = "Weight decay for AdamW optimizer (default 0.1)", value = args.weight_decay, lines = 1, max_lines = 1, elem_id = "pezdispenser_opt_weight_decay")
                            with gr.Column():
                                opt_prompt_bs = gr.Textbox(label = "Number of initializations (default 1)", value = args.prompt_bs, lines = 1, max_lines = 1, elem_id = "pezdispenser_opt_prompt_bs")
                                opt_batch_size = gr.Textbox(label = "Number of target images/prompts used for each iteration (default 1)", value = args.batch_size, lines = 1, max_lines = 1, elem_id = "pezdispenser_opt_prompt_batch_size")

                with gr.Row():
                    gr.HTML(VERSION_HTML)
            
            with gr.Column():
                with gr.Group(elem_id = "pezdispenser_results_column"):
                    with gr.Row():
                        output_prompt = gr.TextArea(label = "Prompt", show_label = True, interactive = False, elem_id = "pezdispenser_output_prompt").style(show_copy_button = True)
                    with gr.Row():
                        with gr.Column():
                            statistics_text = gr.HTML(elem_id = "pezdispenser_statistics_text")
                        with gr.Column():
                            send_to_txt2img_button = gr.Button("Send to txt2img", elem_id = "pezdispenser_send_to_txt2img_button")
                            setattr(send_to_txt2img_button, "do_not_save_to_config", True)
                        with gr.Column():
                            send_to_img2img_button = gr.Button("Send to img2img", elem_id = "pezdispenser_send_to_img2img_button")
                            setattr(send_to_img2img_button, "do_not_save_to_config", True)
                    with gr.Row():
                        interrupt_button = gr.Button("Interrupt", variant = "stop", elem_id = "pezdispenser_interrupt_button", visible = False)

        unload_model_button.click(
            unload_model,
            outputs = [ statistics_text ]
        )

        process_image_button.click(
            ui.wrap_queued_call(inference_image),
            _js = "pezdispenser_submit",
            inputs = [input_text, opt_model, opt_device, opt_prompt_length, opt_num_step, opt_lr, opt_weight_decay, opt_prompt_bs, opt_batch_size] + input_images,
            outputs = [output_prompt, statistics_text]
        )
        process_text_button.click(
            ui.wrap_queued_call(inference_text),
            _js = "pezdispenser_submit",
            inputs = [input_text, opt_model, opt_device, opt_prompt_length, opt_num_step, opt_lr, opt_weight_decay, opt_prompt_bs, opt_batch_size, input_text],
            outputs = [output_prompt, statistics_text]
        )
        interrupt_button.click(interrupt)

        send_to_txt2img_button.click(
            lambda s: s,
            _js = "pezdispenser_switch_to_txt2img",
            inputs = [output_prompt],
            outputs = [find_prompt(ui.txt2img_paste_fields)]
        )

        send_to_img2img_button.click(
            lambda s: s,
            _js = "pezdispenser_switch_to_img2img",
            inputs = [output_prompt],
            outputs = [find_prompt(ui.img2img_paste_fields)]
        )

    for obj in [
        opt_model,
        opt_device,
        opt_prompt_length,
        opt_num_step,
        opt_lr,
        opt_weight_decay,
        opt_prompt_bs,
        opt_batch_size,
        input_text
    ] + input_images:
        setattr(obj, "do_not_save_to_config", True)

    return [(tab, "PEZ Dispenser", "pezdispenser")]


def create_tab_not_installed():
    with gr.Blocks(analytics_enabled = False) as tab:
        gr.Markdown("# Some required packages are not installed.")
        gr.Markdown("## Please restart WebUI to install them automatically.")

    return [(tab, "PEZ Dispenser", "pezdispenser")]


def on_ui_settings():
    section = ("pezdispenser", "PEZ Dispenser")
    shared.opts.add_option("pezdispenser_ui_mode", shared.OptionInfo(
        "Tab and Script",
        "PEZ Dispenser mode (requires restart)",
        gr.Radio,
        lambda: { "choices": [ "Tab and Script", "Tab only", "Script only" ] },
        section = section
    ))


def on_unload():
    this.reset()


if show_tab():
    if state.installed:
        script_callbacks.on_ui_tabs(create_tab)
    else:
        script_callbacks.on_ui_tabs(create_tab_not_installed)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)

########## Script ##########

class ScriptRunHandler:
    def __init__(self, p, parsed_extra_networks):

        self.p = p
        self.parsed_extra_networks = parsed_extra_networks

        self.res_images = []
        self.res_all_prompts = []
        self.res_infotexts = []
        self.res_info = None
        
        self.sample_every_iteration = 0

    def run(self, prompt):
        if shared.state.interrupted:
            return False

        pc = copy.copy(self.p)
        pc.n_iter = 1
        pc.prompt = prompt + self.parsed_extra_networks
        pi = process_images(pc)

        if shared.state.interrupted or pi.images is None:
            return False

        self.res_images += pi.images
        self.res_all_prompts += pi.all_prompts
        self.res_infotexts += pi.infotexts
        if self.res_info is None:
            self.res_info = pi.info

        return True


def on_script_progress(step, total, prompt, run_handler):
    if step == 0:
        this.progress_started = time.time()
    progress = step * 100 // total
    processing_time = time.time() - this.progress_started
    speed = (step / processing_time) if processing_time > 0 else 0
    shared.state.textinfo = f"{this.progress_title} {progress}% ..."
    print(f"\r{this.progress_title}: {progress}% ({speed:.2f}it/s)", end = "", flush = True)
    
    if (run_handler.sample_every_iteration > 0) and (step > 0) and (step < total) and (step % run_handler.sample_every_iteration == 0):
        run_handler.run(prompt)


class Script(scripts.Script):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def title(self):
        return "PEZ Dispenser"

    def show(self, is_img2img):
        if not show_script():
            return False
        if not state.installed:
            return False
        return not is_img2img

    def ui(self, is_img2img):
        if not show_script():
            return []
        if not state.installed:
            return []
        if is_img2img:
            return []
            
        input_images = list()

        gr.HTML('<br />')
        with gr.Row():
            input_type = gr.Radio(label = 'Source', show_label = False,
                choices = [ "Long prompt to short prompt", "Image to prompt" ], value = "Long prompt to short prompt", elem_id = "pezdispenser_script_input_type")
        with gr.Row(elem_id = "pezdispenser_script_input_images_group", visible = False):
            for i in range(1, INPUT_IMAGES_COUNT + 1):
                with gr.Tab(f"Image {i}"):
                    input_images.append(gr.Image(type = "pil", label = "Target image", show_label = False, elem_id = f"pezdispenser_script_input_image_{i}"))

        gr.HTML('<br />')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    opt_model = gr.Dropdown(label = "Model", choices = [n for n, _, _ in pretrained_models], type = "index",
                        value = pretrained_models[this.model_index][0], elem_id = "pezdispenser_script_opt_model")
                    unload_model_button = ToolButton(unload_symbol, elem_id = "pezdispenser_script_unload_model_button")
                    setattr(unload_model_button, "do_not_save_to_config", True)

        gr.HTML('<br />')
        with gr.Row():
            with gr.Column():
                opt_prompt_length = gr.Slider(label = "Prompt length (optimal 8-16)", minimum = 1, maximum = 75, step = 1, value = args.prompt_len, elem_id = "pezdispenser_script_opt_prompt_length")
            with gr.Column():
                opt_num_step = gr.Slider(label = "Optimization steps (optimal 1000-3000)", minimum = 1, maximum = 10000, step = 1, value = args.iter, elem_id = "pezdispenser_script_opt_num_step")
            with gr.Column():
                opt_sample_step = gr.Slider(label = "Sample on every step (0 - disabled)", minimum = 0, maximum = 10000, step = 1, value = 0, elem_id = "pezdispenser_script_opt_sample_step")

        with gr.Row():
            with gr.Accordion("Advanced", open = False):
                with gr.Row():
                    with gr.Column():
                        opt_lr = gr.Textbox(label = "Learning rate for AdamW optimizer (default 0.1)", value = args.lr, lines = 1, max_lines = 1, elem_id = "pezdispenser_script__opt_lr")
                        opt_weight_decay = gr.Textbox(label = "Weight decay for AdamW optimizer (default 0.1)", value = args.weight_decay, lines = 1, max_lines = 1, elem_id = "pezdispenser_script__opt_weight_decay")
                    with gr.Column():
                        opt_prompt_bs = gr.Textbox(label = "Number of initializations (default 1)", value = args.prompt_bs, lines = 1, max_lines = 1, elem_id = "pezdispenser_script__opt_prompt_bs")
                        opt_batch_size = gr.Textbox(label = "Number of target images/prompts used for each iteration (default 1)", value = args.batch_size, lines = 1, max_lines = 1, elem_id = "pezdispenser_script__opt_prompt_batch_size")

        with gr.Row():
            gr.HTML(f"<br/>" + VERSION_HTML)

        input_type.change(
            fn = None,
            _js = "pezdispenser_show_script_images",
            inputs = [input_type]
        )

        unload_model_button.click(
            unload_model
        )

        res = [
            input_type,
            opt_model,
            opt_prompt_length,
            opt_num_step,
            opt_sample_step,
            opt_lr,
            opt_weight_decay,
            opt_prompt_bs,
            opt_batch_size
        ] + input_images

        for obj in res:
            setattr(obj, "do_not_save_to_config", True)

        return res

    def run(self, p,
        input_type,
        model_index,
        prompt_length,
        iterations_count,
        sample_every_iteration,
        lr,
        weight_decay,
        prompt_bs,
        batch_size,
        
        *args
    ):
        if not show_script():
            return Processed(p, [])

        script_name = os.path.splitext(os.path.basename(self.filename))[0]

        if not state.installed:
            raise ModuleNotFoundError("Some required packages are not installed")
        
        input_images = list(filter(lambda img: not img is None, args[0:INPUT_IMAGES_COUNT]))

        is_image = input_type == "Image to prompt"
        parsed_prompts, parsed_extra_networks = parse_prompt(None if is_image else p.prompt)

        if is_image:
            if len(input_images) == 0:
                raise ValueError("Input image is empty")
        else:
            if parsed_prompts is None:
                raise RuntimeError("Prompt is empty")
                
        iterations_count_norm = int(iterations_count) if iterations_count is not None else 1000
        prompt_len = min(int(prompt_length), 75) if prompt_length is not None else 8

        shared.state.job_count = p.n_iter

        saved_textinfo = shared.state.textinfo
        shared.state.textinfo = "Loading model..."
        device_name = devices.get_optimal_device_name()
        model, preprocess, clip_model = load_model(model_index, device_name)
        shared.state.textinfo = saved_textinfo

        run_handler = ScriptRunHandler(p, parsed_extra_networks)

        progress_steps = [ max(iterations_count_norm // 100, 10) ]
        if (sample_every_iteration > 0) and (sample_every_iteration < iterations_count_norm):
            progress_steps.append(sample_every_iteration)
            run_handler.sample_every_iteration = sample_every_iteration
            shared.state.job_count = p.n_iter * (((iterations_count_norm - 1) // sample_every_iteration) + 1)

        for iteration in range(p.n_iter):
            if shared.state.interrupted:
                break

            this.start_progress("Processing image" if is_image else "Processing prompt")
            saved_textinfo = shared.state.textinfo
            shared.state.textinfo = this.progress_title + "..."
            
            optimized_prompt = optimize_prompt(
                model,
                preprocess,
                torch.device(device_name),
                clip_model,
                prompt_len,
                iterations_count_norm,
                float(lr),
                float(weight_decay),
                int(prompt_bs),
                None,
                int(batch_size),
                target_images = input_images if is_image else None,
                target_prompts = None if is_image else parsed_prompts,
                on_progress = on_script_progress,
                progress_steps = progress_steps,
                progress_args = run_handler
            )

            shared.state.textinfo = saved_textinfo

            res = run_handler.run(optimized_prompt)

            if not res:
                break

        return Processed(
            p,
            run_handler.res_images,
            seed = p.seed,
            info = "" if run_handler.res_info is None else run_handler.res_info,
            all_prompts = run_handler.res_all_prompts,
            infotexts = run_handler.res_infotexts
        )
