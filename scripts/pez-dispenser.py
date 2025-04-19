import copy
import os
import json
import time
import re
import argparse
import traceback
import torch
import open_clip
import gc
import gradio as gr
import scripts.optim_utils as utils
from modules import devices, scripts, script_callbacks, ui, shared, progress, extra_networks, patches
from modules.processing import process_images, Processed
from modules.ui_components import ToolButton
from PIL import Image


VERSION = "1.6.2"

ALLOW_DEVICE_SELECTION = False
INPUT_IMAGES_COUNT = 5

unload_symbol = '\u274c' # delete
#unload_symbol = '\u267b' # recycle

VERSION_HTML = f'<table width="100%"><tr><td width="50%">Version <a href="https://github.com/r0mar0ma/sd-webui-pez-dispenser/blob/main/CHANGELOG.md" target="_blank">{VERSION}</a></td><td width="50%" style="text-align: end;"><a href="https://arxiv.org/abs/2302.03668" target="_blank">Hard Prompts Made Easy</a> documentation</td></tr></table>'

class ThisState:

    def __init__(self):
        self.reset()

    def start_progress(self, title):
        self.progress_title = title
        self.progress_started = time.perf_counter()

    def reset(self):
        self.model_index = 0
        self.model_device_name = devices.get_optimal_device_name()
        self.model = None
        self.preprocess = None
        self.precision = "fp32"
        self.clip_model = None
        self.progress_title = ""
        self.progress_started = time.perf_counter()

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
    "device": null,
    "sample_on_iter": 0,
    "dont_sample_repetitive": false,
    "optimizer": "AdamW",
    "torch_compile_level": 0,
    "precision": "fp32",
    "debug": false
}
"""))
if os.path.isfile(config_file_path):
    with open(config_file_path, encoding='utf-8') as f:
        args.__dict__.update(json.load(f))

DEBUG = args.debug

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

########## Utils ##########

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

def optimize_prompt(*args, **kwargs):
    optimize_started = time.perf_counter()
    res = utils.optimize_prompt(*args, **kwargs)
    optimize_time = time.perf_counter() - optimize_started
    if DEBUG:
        cols, _ = os.get_terminal_size()
        print("\r", end = "", flush = True)
        print(" " * cols, end = "", flush = True)
        print(f"\rOptimized in {optimize_time:.3f} sec", flush = True)
    gc_collect()
    return res

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

if not "metaclip_fullcc" in open_clip.pretrained._VITbigG14:
    open_clip.pretrained._VITbigG14["metaclip_fullcc"] = open_clip.pretrained._pcfg(
        #url='https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt',
        hf_hub="timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b/",
        #quick_gelu=True,
    )


allowed_models = [
    r"^RN[0-9]+",
    r"^ViT-",
    r"^convnext_base",
    r"^convnext_large"
]

pretrained_models = [
    ("SD 1.5 (ViT-L-14, openai)", "ViT-L-14", "openai"),
    ("SD 2.0, Midjourney (ViT-H-14, laion2b_s32b_b79k)", "ViT-H-14", "laion2b_s32b_b79k"),
    ("SDXL 1.0 (ViT-bigG-14, laion2b_s39b_b160k)", "ViT-bigG-14", "laion2b_s39b_b160k")
]
for m, p in open_clip.pretrained.list_pretrained(as_str = False):
    for r in allowed_models:
        if re.match(r, m, re.I):
            pretrained_models.append((f"{m}, {p}", m, p))
            break

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

    gc_collect()

    if not this.model is None:
        del this.model
        this.model = None
    if not this.preprocess is None:
        del this.preprocess
        this.preprocess = None
    this.precision = "fp32";
    
    gc_collect()

    if is_cpu:
        msg = "Model unloaded"
    else:
        memory_used = torch.cuda.memory_allocated(device)
        memory_freed = memory_used_pre - memory_used
        msg = f"Model unloaded, GPU memory freed: {(memory_freed / 1048576):.2f} MB"
        if DEBUG:
            msg += f", total GPU memory used: {(memory_used / 1048576):.2f} MB"

    print(msg)
    return msg


def load_model(index, device_name, precision):
    if this.model is None or this.preprocess is None or this.model_index != index or this.model_device_name != device_name or this.precision != precision:
        _, clip_model, clip_pretrain = pretrained_models[index];

        unload_model()

        print(f"Loading model: {clip_model}:{clip_pretrain}, device: {get_device_display_name(device_name)}")

        gc_collect()
        
        is_cpu = device_name == "cpu"
        device = torch.device(device_name)
        if not is_cpu:
            memory_used_pre = torch.cuda.memory_allocated(device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained = clip_pretrain,
            device = device,
            precision = precision
        )

        this.model = model
        this.preprocess = preprocess
        this.precision = precision
        this.clip_model = clip_model
        this.model_index = index
        this.model_device_name = device_name

        gc_collect()

        if is_cpu:
            print("Model loaded")
        else:
            memory_used = torch.cuda.memory_allocated(device)
            memory_taken = memory_used - memory_used_pre
            msg = f"Model loaded, GPU memory taken: {(memory_taken / 1048576):.2f} MB"
            if DEBUG:
                msg += f", total GPU memory used: {(memory_used / 1048576):.2f} MB"
            print(msg)

    return this.model, this.preprocess, this.clip_model

def on_ui_reload():
    utils.reset_forward_text_embedding_compiled()
    unload_model()

script_callbacks.on_before_reload(on_ui_reload)

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

re_normalize_result = [
    (re.compile(r"""[()[\]{}<>\\|/`~!?@#$%^&*+=:;"'\t\r\n]"""), " "),
    #(re.compile(r"""\s[.-]+\s"""), " "),
    (re.compile(r"""\s+"""), " ")
]

def normalize_result(prompt):
    res = prompt
    for r, s in re_normalize_result:
        res = re.sub(r, s, res)
    return res.strip()

def on_progress(step, total, prompt, progress_args):
    if step == 0:
        this.progress_started = time.perf_counter()

    progress = step * 100 // total

    processing_time = time.perf_counter() - this.progress_started
    speed = (step / processing_time) if processing_time > 0 else 0

    print(f"\r{this.progress_title}: {progress}% ({speed:.2f}it/s)", end = "", flush = True)

    shared.state.job_no = step - 1
    shared.state.nextjob()

def inference(
    task_id, 
    model_index, 
    device_name_index, 
    prompt_length, 
    iterations_count, 
    lr, 
    weight_decay, 
    prompt_bs, 
    batch_size, 
    optimizer,
    torch_compile_level,
    precision,
    target_images = None, 
    target_prompt = None
):
    progress.add_task_to_queue(task_id)
    shared.state.begin()
    progress.start_task(task_id)
    shared.state.textinfo = "Preparing..."
    shared.state.job_count = iterations_count

    res = "", ""

    try:
        if not utils.state.installed:
            raise ModuleNotFoundError("Some required packages are not installed. Please restart WebUI to install them automatically.")

        parsed_prompts, parsed_extra_networks = parse_prompt(target_prompt)
        parsed_images = list() if target_images is None else list(filter(lambda i: not i is None, target_images))
        #print(parsed_prompts)
        #print(parsed_images)

        if len(parsed_images) == 0 and parsed_prompts is None:
            raise ValueError("Nothing to process")

        device_name = available_devices[device_name_index][0] if ALLOW_DEVICE_SELECTION else this.model_device_name
        device = torch.device(device_name)

        shared.state.textinfo = "Loading model..."
        model, preprocess, clip_model = load_model(model_index, device_name, precision)

        if torch_compile_level > 0:
            shared.state.textinfo = "Warm up model..."
            if DEBUG:
                print("Warm up model with short optimization cycle")
            optimize_prompt(
                model,
                preprocess,
                device,
                clip_model,
                5,
                2,
                float(lr),
                float(weight_decay),
                1,
                None,
                1,
                target_images = None,
                target_prompts = ["sample"],
                optimizer = list(utils.optimizers)[optimizer],
                torch_compile_level = torch_compile_level
            )

        prompt_len = min(int(prompt_length), 75) if prompt_length is not None else 8
        iter = int(iterations_count) if iterations_count is not None else 1000

        shared.state.textinfo = "Processing..."

        optimized_prompt = optimize_prompt(
            model,
            preprocess,
            device,
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
            optimizer = list(utils.optimizers)[optimizer],
            torch_compile_level = torch_compile_level,
            on_progress = on_progress,
            progress_steps = [ max(iter // 100, 10) ]
        )
        optimized_prompt = normalize_result(optimized_prompt)

        print("")
        processing_time = time.perf_counter() - shared.state.time_start
        res = optimized_prompt + parsed_extra_networks, f"Time taken: {processing_time:.2f} sec"

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

def inference_image(
    task_id, 
    model_index, 
    device_name_index, 
    prompt_length, 
    iterations_count, 
    lr, 
    weight_decay, 
    prompt_bs, 
    batch_size, 
    optimizer,
    torch_compile_level,
    precision,
    *target_images
):
    this.start_progress("Processing image")
    return inference(
        task_id, 
        model_index, 
        device_name_index, 
        prompt_length, 
        iterations_count, 
        lr, 
        weight_decay, 
        prompt_bs, 
        batch_size, 
        optimizer,
        torch_compile_level,
        precision,
        target_images = target_images
    )

def inference_text(
    task_id, 
    model_index, 
    device_name_index, 
    prompt_length, 
    iterations_count, 
    lr, 
    weight_decay, 
    prompt_bs, 
    batch_size, 
    optimizer,
    torch_compile_level,
    precision,
    target_prompt
):
    this.start_progress("Processing prompt")
    return inference(
        task_id, 
        model_index, 
        device_name_index, 
        prompt_length, 
        iterations_count, 
        lr, 
        weight_decay, 
        prompt_bs, 
        batch_size, 
        optimizer,
        torch_compile_level,
        precision,
        target_prompt = target_prompt if not target_prompt is None and target_prompt != "" else None
    )

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
                        for i in range(INPUT_IMAGES_COUNT):
                            with gr.Tab("Image" if i == 0 else f"Extra image {i}"):
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
                        opt_prompt_length = gr.Slider(label = "Prompt length (optimal 8-16)", minimum = 1, maximum = 75, step = 1, 
                            value = args.prompt_len, elem_id = "pezdispenser_opt_prompt_length")
                    with gr.Column():
                        opt_num_step = gr.Slider(label = "Optimization steps (optimal 1000-3000)", minimum = 1, maximum = 10000, step = 1, 
                            value = args.iter, elem_id = "pezdispenser_opt_num_step")

                with gr.Row():
                    with gr.Accordion("Advanced", open = False):
                        with gr.Row(variant="compact"):
                            with gr.Column():
                                opt_lr = gr.Textbox(label = "Learning rate for optimizer (default 0.1)", value = args.lr, lines = 1, 
                                    max_lines = 1, elem_id = "pezdispenser_opt_lr")
                                opt_weight_decay = gr.Textbox(label = "Weight decay for optimizer (default 0.1)", value = args.weight_decay, 
                                    lines = 1, max_lines = 1, elem_id = "pezdispenser_opt_weight_decay")
                            with gr.Column():
                                opt_prompt_bs = gr.Textbox(label = "Number of initializations (default 1)", value = args.prompt_bs, 
                                    lines = 1, max_lines = 1, elem_id = "pezdispenser_opt_prompt_bs")
                                opt_batch_size = gr.Textbox(label = "Number of target images/prompts used for each iteration (default 1)", 
                                    value = args.batch_size, lines = 1, max_lines = 1, elem_id = "pezdispenser_opt_prompt_batch_size")

                with gr.Row():
                    with gr.Accordion("Experimental", open = False):
                        with gr.Row(variant="compact"):
                            with gr.Column():
                                optimizers_names = [ f"{n} ({d})" for n, d in utils.optimizers.items() ]
                                args_optimizer = optimizers_names[list(utils.optimizers).index(args.optimizer)] if args.optimizer in utils.optimizers else optimizers_names[0]
                                opt_optimizer = gr.Dropdown(label = "Optimizer", choices = optimizers_names, type = "index", 
                                    value = args_optimizer, elem_id = "pezdispenser_opt_optimizer")
                                opt_torch_compile_level = gr.Slider(label = "Torch compilation level", 
                                    minimum = utils.TORCH_COMPILE_LEVEL_OFF, maximum = utils.TORCH_COMPILE_LEVEL_MAX, step = 1, 
                                    value = args.torch_compile_level, elem_id = "pezdispenser_opt_torch_compile_level")
                                opt_precision = gr.Dropdown(label = "Precision", choices = ["fp32", "fp16", "bf16"], 
                                    value = args.precision, elem_id = "pezdispenser_opt_precision")

                with gr.Row():
                    gr.HTML(VERSION_HTML)
            
            with gr.Column():
                with gr.Group(elem_id = "pezdispenser_results_column"):
                    with gr.Row():
                        output_prompt = gr.TextArea(label = "Prompt", show_label = True, interactive = False, show_copy_button = True, elem_id = "pezdispenser_output_prompt")
                    with gr.Row():
                        statistics_text = gr.HTML(elem_id = "pezdispenser_statistics_text")
                    with gr.Row():
                        with gr.Column(min_width = 50):
                            send_to_txt2img_button = gr.Button("Send to txt2img", elem_id = "pezdispenser_send_to_txt2img_button")
                            setattr(send_to_txt2img_button, "do_not_save_to_config", True)
                        with gr.Column(min_width = 50):
                            send_to_img2img_button = gr.Button("Send to img2img", elem_id = "pezdispenser_send_to_img2img_button")
                            setattr(send_to_img2img_button, "do_not_save_to_config", True)
                    with gr.Row():
                        interrupt_button = gr.Button("Interrupt", variant = "stop", elem_id = "pezdispenser_interrupt_button", visible = False)

        def unload_model_button_click():
            res = unload_model()
            try:
                gr.Info(res)
                return ""
            except:
                return res
        unload_model_button.click(
            unload_model_button_click,
            outputs = [ statistics_text ]
        )

        process_image_button.click(
            ui.wrap_queued_call(inference_image),
            _js = "pezdispenser_submit",
            inputs = [
                input_text, 
                opt_model, 
                opt_device, 
                opt_prompt_length, 
                opt_num_step, 
                opt_lr, 
                opt_weight_decay, 
                opt_prompt_bs, 
                opt_batch_size,
                opt_optimizer,
                opt_torch_compile_level,
                opt_precision
            ] + input_images,
            outputs = [output_prompt, statistics_text]
        )
        process_text_button.click(
            ui.wrap_queued_call(inference_text),
            _js = "pezdispenser_submit",
            inputs = [
                input_text, 
                opt_model, 
                opt_device, 
                opt_prompt_length, 
                opt_num_step, 
                opt_lr, 
                opt_weight_decay, 
                opt_prompt_bs, 
                opt_batch_size, 
                opt_optimizer,
                opt_torch_compile_level,
                opt_precision,
                input_text
            ],
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
        opt_optimizer,
        opt_torch_compile_level,
        opt_precision,
        input_text,
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
    if utils.state.installed:
        script_callbacks.on_ui_tabs(create_tab)
    else:
        script_callbacks.on_ui_tabs(create_tab_not_installed)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)

########## Script ##########

class ScriptRunHandler:
    def __init__(self, p, dont_sample_repetitive):

        self.p = p

        self.res_images = []
        self.res_all_prompts = []
        self.res_infotexts = []
        self.res_info = None
        
        self.sample_every_iteration = 0
        self.extra_networks = None

        self.last_run_prompt = ""
        self.dont_sample_repetitive = dont_sample_repetitive

    def run(self, prompt):
        if shared.state.interrupted:
            return False

        run_prompt = normalize_result(prompt) + (self.extra_networks or "")

        if self.dont_sample_repetitive and self.last_run_prompt == run_prompt:
            cols, _ = os.get_terminal_size()
            print("\r", end = "", flush = True)
            print(" " * cols, end = "", flush = True)
            print("\rSkipping repetitive sample", flush = True)
            shared.state.nextjob()
            return True
        self.last_run_prompt = run_prompt

        pc = copy.copy(self.p)
        pc.n_iter = 1
        pc.prompt = run_prompt
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
        this.progress_started = time.perf_counter()
    progress = step * 100 // total
    processing_time = time.perf_counter() - this.progress_started
    speed = (step / processing_time) if processing_time > 0 else 0
    shared.state.textinfo = f"{this.progress_title} {progress}% ..."
    print(f"\r{this.progress_title}: {progress}% ({speed:.2f}it/s)", end = "", flush = True)
    
    if (run_handler.sample_every_iteration > 0) and (step > 0) and (step < total) and (step % run_handler.sample_every_iteration == 0):
        run_handler.run(prompt)


VALUE_TYPE_PROMPT = "Long prompt to short prompt"
VALUE_TYPE_IMAGE = "Image to prompt"
VALUE_TYPE_IMAGES_BATCH = "Batch images to prompt"

class Script(scripts.Script):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def title(self):
        return "PEZ Dispenser"

    def show(self, is_img2img):
        if not show_script():
            return False
        if not utils.state.installed:
            return False
        return not is_img2img

    def ui(self, is_img2img):
        if not show_script():
            return []
        if not utils.state.installed:
            return []
        if is_img2img:
            return []
            
        input_images = list()
        input_batch_images = list()

        gr.HTML("<br />")
        with gr.Row():
            input_type = gr.Radio(label = 'Source', show_label = False,
                choices = [ VALUE_TYPE_PROMPT, VALUE_TYPE_IMAGE, VALUE_TYPE_IMAGES_BATCH ], value = VALUE_TYPE_PROMPT, 
                elem_id = "pezdispenser_script_input_type")

        with gr.Row(elem_id = "pezdispenser_script_input_prompt_group", visible = True) as input_prompt_group:
            input_prompt_split_prompt = gr.Checkbox(label = 'Split prompt by lines', value = False, 
                elem_id = "pezdispenser_script_input_prompt_split_prompt")

        with gr.Row(elem_id = "pezdispenser_script_input_images_group", visible = False) as input_images_group:
            with gr.Column():
                with gr.Row():
                    for i in range(INPUT_IMAGES_COUNT):
                        with gr.Tab("Image" if i == 0 else f"Extra image {i}"):
                            input_images.append(gr.Image(type = "pil", show_label = False, elem_id = f"pezdispenser_script_input_image_{i}"))
                with gr.Row():
                    input_images_use_prompt = gr.Checkbox(label = 'Use prompt', value = False, 
                        elem_id = "pezdispenser_script_input_images_use_prompt")
                    input_images_split_prompt = gr.Checkbox(label = 'Split prompt by lines', value = False, 
                        elem_id = "pezdispenser_script_input_images_split_prompt")

        with gr.Row(elem_id = "pezdispenser_script_input_images_batch_group", visible = False) as input_images_batch_group:
            with gr.Column():
                with gr.Row():
                    input_batch_folder = gr.Textbox(label = "Input directory", lines = 1, max_lines = 1, 
                        elem_id = "pezdispenser_script_input_batch_folder")
                with gr.Row():
                    with gr.Accordion("Extra images", open = False):
                        for i in range(1, INPUT_IMAGES_COUNT):
                            with gr.Tab(f"Extra image {i}"):
                                input_batch_images.append(gr.Image(type = "pil", show_label = False, 
                                elem_id = f"pezdispenser_script_input_batch_image_{i}"))
                with gr.Row():
                    input_batch_images_use_prompt = gr.Checkbox(label = 'Use prompt', value = False, 
                        elem_id = "pezdispenser_script_input_batch_images_use_prompt")
                    input_batch_images_split_prompt = gr.Checkbox(label = 'Split prompt by lines', 
                        value = False, elem_id = "pezdispenser_script_input_batch_images_split_prompt")

        gr.HTML("<br />")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    opt_model = gr.Dropdown(label = "Model", choices = [n for n, _, _ in pretrained_models], type = "index",
                        value = pretrained_models[this.model_index][0], elem_id = "pezdispenser_script_opt_model")
                    unload_model_button = ToolButton(unload_symbol, elem_id = "pezdispenser_script_unload_model_button")
                    setattr(unload_model_button, "do_not_save_to_config", True)

        gr.HTML("<br />")
        with gr.Row(variant="compact"):
            with gr.Column():
                opt_prompt_length = gr.Slider(label = "Prompt length (optimal 8-16)", minimum = 1, maximum = 75, step = 1, 
                    value = args.prompt_len, elem_id = "pezdispenser_script_opt_prompt_length")
                opt_num_step = gr.Slider(label = "Optimization steps (optimal 1000-3000)", minimum = 1, maximum = 10000, step = 1, 
                    value = args.iter, elem_id = "pezdispenser_script_opt_num_step")
            with gr.Column():
                opt_sample_step = gr.Slider(label = "Sample on every step (0 - disabled)", minimum = 0, maximum = 10000, step = 1, 
                    value = args.sample_on_iter, elem_id = "pezdispenser_script_opt_sample_step")
                opt_sample_repetitive = gr.Checkbox(label = "Do not make repetitive samples", value = args.dont_sample_repetitive, 
                    elem_id = "pezdispenser_script_opt_sample_repetitive")

        with gr.Row():
            with gr.Accordion("Advanced", open = False):
                with gr.Row(variant="compact"):
                    with gr.Column():
                        opt_lr = gr.Textbox(label = "Learning rate for optimizer (default 0.1)", value = args.lr, lines = 1, 
                            max_lines = 1, elem_id = "pezdispenser_script_opt_lr")
                        opt_weight_decay = gr.Textbox(label = "Weight decay for optimizer (default 0.1)", value = args.weight_decay, 
                            lines = 1, max_lines = 1, elem_id = "pezdispenser_script_opt_weight_decay")
                    with gr.Column():
                        opt_prompt_bs = gr.Textbox(label = "Number of initializations (default 1)", value = args.prompt_bs, lines = 1, 
                            max_lines = 1, elem_id = "pezdispenser_script_opt_prompt_bs")
                        opt_batch_size = gr.Textbox(label = "Number of target images/prompts used for each iteration (default 1)", 
                            value = args.batch_size, lines = 1, max_lines = 1, elem_id = "pezdispenser_script_opt_prompt_batch_size")

        with gr.Row():
            with gr.Accordion("Experimental", open = False):
                with gr.Row(variant="compact"):
                    with gr.Column():
                        optimizers_names = [ f"{n} ({d})" for n, d in utils.optimizers.items() ]
                        args_optimizer = optimizers_names[list(utils.optimizers).index(args.optimizer)] if args.optimizer in utils.optimizers else optimizers_names[0]
                        opt_optimizer = gr.Dropdown(label = "Optimizer", choices = optimizers_names, type = "index", 
                            value = args_optimizer, elem_id = "pezdispenser_script_opt_optimizer")
                        opt_torch_compile_level = gr.Slider(label = "Torch compilation level", 
                            minimum = utils.TORCH_COMPILE_LEVEL_OFF, maximum = utils.TORCH_COMPILE_LEVEL_MAX, step = 1, 
                            value = args.torch_compile_level, elem_id = "pezdispenser_script_opt_torch_compile_level")
                        opt_precision = gr.Dropdown(label = "Precision", choices = ["fp32", "fp16", "bf16"], 
                            value = args.precision, elem_id = "pezdispenser_script_opt_precision")

        with gr.Row():
            gr.HTML(f"<br />" + VERSION_HTML)

        def input_type_change(t):
            input_prompt_group.visible = (t == VALUE_TYPE_PROMPT)
            input_images_group.visible = (t == VALUE_TYPE_IMAGE)
            input_images_batch_group.visible = (t == VALUE_TYPE_IMAGES_BATCH)
            return [
                gr.Row.update(visible = input_prompt_group.visible),
                gr.Row.update(visible = input_images_group.visible),
                gr.Row.update(visible = input_images_batch_group.visible),
            ]
        input_type.change(
            fn = input_type_change,
            inputs = [input_type],
            outputs = [
                input_prompt_group,
                input_images_group,
                input_images_batch_group
            ]
        )

        def unload_model_button_click():
            res = unload_model()
            try:
                gr.Info(res)
                return ""
            except:
                return res
        unload_model_button.click(
            unload_model_button_click
        )

        res = [
            input_type,
            input_prompt_split_prompt,
            input_images_use_prompt,
            input_images_split_prompt,
            input_batch_images_use_prompt,
            input_batch_images_split_prompt,
            input_batch_folder,
            opt_model,
            opt_prompt_length,
            opt_num_step,
            opt_sample_step,
            opt_sample_repetitive,
            opt_lr,
            opt_weight_decay,
            opt_prompt_bs,
            opt_batch_size,
            opt_optimizer,
            opt_torch_compile_level,
            opt_precision
        ] + input_images + input_batch_images

        for obj in res:
            setattr(obj, "do_not_save_to_config", True)

        return res

    def run(self, p,
        input_type,
        input_prompt_split_prompt,
        input_images_use_prompt,
        input_images_split_prompt,
        input_batch_images_use_prompt,
        input_batch_images_split_prompt,
        input_batch_folder,
        model_index,
        prompt_length,
        iterations_count,
        sample_every_iteration,
        dont_sample_repetitive,
        lr,
        weight_decay,
        prompt_bs,
        batch_size,
        optimizer,
        torch_compile_level,
        precision,
        
        *args
    ):
        if not show_script():
            return Processed(p, [])

        script_name = os.path.splitext(os.path.basename(self.filename))[0]

        if not utils.state.installed:
            raise ModuleNotFoundError("Some required packages are not installed")

        if (optimizer < 0) or (optimizer >= len(utils.optimizers)):
            raise ValueError("Unknown optimizer")

        if input_type == VALUE_TYPE_PROMPT:
            split_prompt = input_prompt_split_prompt
        elif input_type == VALUE_TYPE_IMAGE:
            split_prompt = input_images_split_prompt
        elif input_type == VALUE_TYPE_IMAGES_BATCH:
            split_prompt = input_batch_images_split_prompt
        else:
            split_prompt = False
        
        parsed_prompts = [parse_prompt(p)
            for p in [ s.strip()
                for s in (p.prompt.rstrip().splitlines() if split_prompt else [ p.prompt.strip() ]) ]]
        
        if input_type == VALUE_TYPE_PROMPT:
            jobs = [(p, e, None, None)
                for p, e in filter(lambda p: not p[0] is None, parsed_prompts)
            ]
            if len(jobs) == 0:
                raise ValueError("Prompt is empty")

        elif input_type == VALUE_TYPE_IMAGE:
            input_images_args = args[0:INPUT_IMAGES_COUNT]
            input_images = list(filter(lambda img: not img is None, input_images_args))

            if input_images is None or len(input_images) == 0:
                raise ValueError("Input image is empty")

            jobs = [(p, e, input_images, None)
                for p, e in (parsed_prompts if input_images_use_prompt and (len(parsed_prompts) > 0) else [(None, None)])
            ]
        
        elif input_type == VALUE_TYPE_IMAGES_BATCH:
            input_batch_images_args = args[INPUT_IMAGES_COUNT:INPUT_IMAGES_COUNT + INPUT_IMAGES_COUNT - 1]
            input_batch_images = list(filter(lambda img: not img is None, input_batch_images_args))
            
            if not os.path.isdir(input_batch_folder):
                raise ValueError("Input directory does not exist")
            input_image_files = sorted(list(filter(lambda f: os.path.isfile(os.path.join(input_batch_folder, f)) and (f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")), os.listdir(input_batch_folder))))
            if input_image_files is None or len(input_image_files) == 0:
                raise ValueError("Input directory has no images")
            
            jobs = [(p, e, input_batch_images, f)
                for f in input_image_files
                for p, e in (parsed_prompts if input_batch_images_use_prompt and (len(parsed_prompts) > 0) else [(None, None)])
            ]
                
        iterations_count_norm = int(iterations_count) if iterations_count is not None else 1000
        prompt_len = min(int(prompt_length), 75) if prompt_length is not None else 8

        shared.state.job_count = len(jobs) * p.n_iter

        saved_textinfo = shared.state.textinfo
        
        shared.state.textinfo = "Loading model..."
        device_name = devices.get_optimal_device_name()
        device = torch.device(device_name)
        is_cpu = device_name == "cpu"
        model, preprocess, clip_model = load_model(model_index, device_name, precision)

        if torch_compile_level > 0:
            shared.state.textinfo = "Warm up model..."
            if DEBUG:
                print("Warm up model with short optimization cycle")
            optimize_prompt(
                model,
                preprocess,
                device,
                clip_model,
                5,
                2,
                float(lr),
                float(weight_decay),
                1,
                None,
                1,
                target_images = None,
                target_prompts = ["sample"],
                optimizer = list(utils.optimizers)[optimizer],
                torch_compile_level = torch_compile_level
            )
        
        shared.state.textinfo = saved_textinfo

        run_handler = ScriptRunHandler(p, dont_sample_repetitive)

        progress_steps = [ max(iterations_count_norm // 100, 10) ]
        if (sample_every_iteration > 0) and (sample_every_iteration < iterations_count_norm):
            progress_steps.append(sample_every_iteration)
            run_handler.sample_every_iteration = sample_every_iteration
            shared.state.job_count *= (((iterations_count_norm - 1) // sample_every_iteration) + 1)

        for prompts, extra_networks, images, image_file in jobs:
            if shared.state.interrupted:
                break

            run_handler.extra_networks = extra_networks

            target_prompts = prompts
            target_images = list()

            if input_type == VALUE_TYPE_PROMPT:
                progress_title = "Processing prompt"
            elif input_type == VALUE_TYPE_IMAGE:
                progress_title = "Processing image"
            elif input_type == VALUE_TYPE_IMAGES_BATCH:
                progress_title = f"Processing file {image_file}"
                try:
                    target_images.append(Image.open(os.path.join(input_batch_folder, image_file)))
                except Exception as ex:
                    print()
                    print(f"{ex.__class__.__name__}: {ex}")
                    continue

            if not images is None:
                target_images.extend([i.copy() for i in images])

            try:
                for iteration in range(p.n_iter):
                    if shared.state.interrupted:
                        break
                    
                    if DEBUG:
                        msg = f"{progress_title}, iteration {iteration + 1} of {p.n_iter}, images: {len(target_images)}, prompts: {target_prompts}"
                        if not is_cpu:
                            memory_used = torch.cuda.memory_allocated(device)
                            msg += f", total GPU memory used: {(memory_used / 1048576):.2f} MB"
                        print(msg)

                    this.start_progress(progress_title)
                    saved_textinfo = shared.state.textinfo
                    shared.state.textinfo = this.progress_title + "..."
                    
                    optimized_prompt = optimize_prompt(
                        model,
                        preprocess,
                        device,
                        clip_model,
                        prompt_len,
                        iterations_count_norm,
                        float(lr),
                        float(weight_decay),
                        int(prompt_bs),
                        None,
                        int(batch_size),
                        target_images = target_images,
                        target_prompts = target_prompts,
                        on_progress = on_script_progress,
                        progress_steps = progress_steps,
                        progress_args = run_handler,
                        optimizer = list(utils.optimizers)[optimizer],
                        torch_compile_level = torch_compile_level
                    )

                    shared.state.textinfo = saved_textinfo

                    if shared.state.interrupted:
                        break
                    if not run_handler.run(optimized_prompt):
                        break

            finally:
                if not target_images is None:
                    for img in target_images:
                        img.close()

        return Processed(
            p,
            run_handler.res_images,
            seed = p.seed,
            info = run_handler.res_info or "",
            all_prompts = run_handler.res_all_prompts,
            infotexts = run_handler.res_infotexts
        )
