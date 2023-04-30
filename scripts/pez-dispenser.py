import os
import json
import time
import argparse
import traceback
import torch
import open_clip
import gradio as gr
from common.optim_utils import optimize_prompt, state
from modules import devices, scripts, script_callbacks, ui, shared, progress

ALLOW_DEVICE_SELECTION = False

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
        self.progress_title = ""
        self.progress_started = time.time()

this = ThisState()

########## Arguments ##########

config_dir = scripts.basedir()
config_file_path = os.path.join(config_dir, "config.json")

args = argparse.Namespace()
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
    with open(config_file_path) as f:
        args.__dict__.update(json.load(f))
args.print_step = None

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

def load_model(index, device_name):
    if this.model is None or this.preprocess is None or this.model_index != index or this.model_device_name != device_name:
        _, clip_model, clip_pretrain = pretrained_models[index];

        print(f"Loading model: {clip_model}:{clip_pretrain}, device: {get_device_display_name(device_name)}")

        model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained = clip_pretrain, device = torch.device(device_name))
        
        this.model = model
        this.preprocess = preprocess
        this.model_index = index
        this.model_device_name = device_name

    return this.model, this.preprocess

########## Processing ##########

def on_progress(step, total):
    if step == 0:
        this.progress_started = time.time()

    progress = step * 100 // total

    processing_time = time.time() - this.progress_started
    speed = (step / processing_time) if processing_time > 0 else 0

    print(f"\r{this.progress_title}: {progress}% ({speed:.2f}it/s)", end = "", flush = True)

    shared.state.job_no = step - 1
    shared.state.nextjob()

def inference(task_id, model_index, device_name_index, prompt_length, iterations_count, target_image = None, target_prompt = None):
    progress.add_task_to_queue(task_id)
    shared.state.begin()
    progress.start_task(task_id)
    shared.state.textinfo = "Preparing..."
    shared.state.job_count = iterations_count

    res = "", ""

    try :
        if not state.installed:
            raise ModuleNotFoundError("Some required packages are not installed. Please restart WebUI to install them automatically.")

        if target_image is None and target_prompt is None:
            raise ValueError("Nothing to process")

        device_name = available_devices[device_name_index][0] if ALLOW_DEVICE_SELECTION else this.model_device_name

        shared.state.textinfo = "Loading model..."
        model, preprocess = load_model(model_index, device_name)
        
        args.prompt_len = min(int(prompt_length), 75) if prompt_length is not None else 8
        args.iter = int(iterations_count) if iterations_count is not None else 1000

        shared.state.textinfo = "Processing..."
        
        prompt = optimize_prompt(
            model,
            preprocess,
            args,
            torch.device(device_name),
            target_images = [target_image] if not target_image is None else None,
            target_prompts = [target_prompt] if not target_prompt is None else None,
            on_progress = on_progress,
            progress_step = max(args.iter // 100, 10)
        )
        processing_time = time.time() - state.time_start

        print("")

        res = prompt, f"Time taken: {processing_time:.2f} sec"

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

def inference_image(task_id, target_image, model_index, device_name_index, prompt_length, iterations_count):
    this.start_progress("Processing image")
    return inference(task_id, model_index, device_name_index, prompt_length, iterations_count, target_image = target_image)
    
def inference_text(task_id, target_prompt, model_index, device_name_index, prompt_length, iterations_count):
    this.start_progress("Processing prompt")
    return inference(task_id, model_index, device_name_index, prompt_length, iterations_count, target_prompt = target_prompt if not target_prompt is None and target_prompt != "" else None)
    
def interrupt():
    state.interrupt()
        
########## App ##########

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

def add_tab():
    with gr.Blocks(analytics_enabled = False) as tab:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Tab("Image to Prompt"):
                        input_image = gr.Image(type = "pil", label = "Target Image", show_label = False, elem_id = "pezdispenser_input_image")
                        process_image_button = gr.Button("Generate Prompt", variant = "primary", elem_id = "pezdispenser_process_image_button")

                    with gr.Tab("Long Prompt to Short Prompt"):
                        input_text = gr.TextArea(label = "Target Prompt", show_label = False, interactive = True, elem_id = "pezdispenser_input_text")
                        process_text_button = gr.Button("Distill Prompt", variant = "primary", elem_id = "pezdispenser_process_text_button")

                with gr.Row():
                    with gr.Column():
                        opt_model = gr.Dropdown(label = "Model", choices = [n for n, _, _ in pretrained_models], type = "index", 
                            value = pretrained_models[this.model_index][0], elem_id = "pezdispenser_opt_model")
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
                        opt_prompt_length = gr.Slider(label = "Prompt Length (optimal 8-16)", minimum = 1, maximum = 75, step = 1, value = args.prompt_len, elem_id = "pezdispenser_opt_prompt_length")
                    with gr.Column():
                        opt_num_step = gr.Slider(label = "Optimization Steps (optimal 1000-3000)", minimum = 1, maximum = 10000, step = 1, value = args.iter, elem_id = "pezdispenser_opt_num_step")

            with gr.Column():
                with gr.Group(elem_id = "pezdispenser_results_column"):
                    with gr.Row():
                        output_prompt = gr.TextArea(label = "Prompt", show_label = True, interactive = False, elem_id = "pezdispenser_output_prompt").style(show_copy_button = True)
                    with gr.Row():
                        with gr.Column():
                            statistics_text = gr.HTML(elem_id = "pezdispenser_statistics_text")
                        with gr.Column():
                            send_to_txt2img_button = gr.Button("Send to txt2img", elem_id = "pezdispenser_send_to_txt2img_button")
                        with gr.Column():
                            send_to_img2img_button = gr.Button("Send to img2img", elem_id = "pezdispenser_send_to_img2img_button")
                    with gr.Row():
                        interrupt_button = gr.Button("Interrupt", variant = "stop", elem_id = "pezdispenser_interrupt_button", visible = False)

        process_image_button.click(
            ui.wrap_queued_call(inference_image),
            _js = "pezdispenser_submit",
            inputs = [input_text, input_image, opt_model, opt_device, opt_prompt_length, opt_num_step],
            outputs = [output_prompt, statistics_text]
        )
        process_text_button.click(
            inference_text,
            _js = "pezdispenser_submit",
            inputs = [input_text, input_text, opt_model, opt_device, opt_prompt_length, opt_num_step],
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

    return [(tab, "PEZ Dispenser", "pezdispenser")]


def add_tab_not_installed():
    with gr.Blocks(analytics_enabled = False) as tab:
        gr.Markdown("# Some required packages are not installed.")
        gr.Markdown("## Please restart WebUI to install them automatically.")

    return [(tab, "PEZ Dispenser", "pezdispenser")]


def on_ui_settings():
    pass
    #section = ("pezdispenser", "PEZ Dispenser")
    #shared.opts.add_option("pezdispenser_stub", shared.OptionInfo("", "Stub", section = section))


def on_unload():
    this.reset()


if state.installed:
    script_callbacks.on_ui_tabs(add_tab)
else:
    script_callbacks.on_ui_tabs(add_tab_not_installed)

#script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)
