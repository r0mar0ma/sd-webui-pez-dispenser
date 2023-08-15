import copy
import os
import json
import time
import argparse
import traceback
import torch
import open_clip
import gradio as gr
import common.optim_utils as utils
from modules import devices, scripts, script_callbacks, ui, shared, progress, extra_networks
from modules.processing import process_images, Processed
from modules.ui_components import ToolButton

ALLOW_DEVICE_SELECTION = False

unload_symbol = '\u274c' # delete
#unload_symbol = '\u267b' # recycle

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
    with open(config_file_path, encoding='utf-8') as f:
        args.__dict__.update(json.load(f))
args.print_step = None

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
        return

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
        print("Model unloaded")
    else:
        memory_freed = memory_used_pre - torch.cuda.memory_allocated(device)
        print(f"Model unloaded, GPU memory freed: {(memory_freed / 1048576):.2f} MB")



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
        this.model_index = index
        this.model_device_name = device_name

        if is_cpu:
            print("Model loaded")
        else:
            memory_taken = torch.cuda.memory_allocated(device) - memory_used_pre
            print(f"Model loaded, GPU memory taken: {(memory_taken / 1048576):.2f} MB")

    return this.model, this.preprocess

########## Processing ##########

def parse_prompt(prompt):
    parsed_prompt = None
    parsed_extra_networks = ""
    if not prompt is None:
        parsed_prompt, extra_network_data = extra_networks.parse_prompt(prompt)
        for extra_network_name, extra_network_args in extra_network_data.items():
            for arg in extra_network_args:
                parsed_extra_networks += f"<{extra_network_name}:{':'.join(arg.items)}>"
        if parsed_prompt.strip() == "":
            parsed_prompt = None
    return parsed_prompt, parsed_extra_networks

def on_progress(step, total, prompt, progress_args):
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

    try:
        if not utils.state.installed:
            raise ModuleNotFoundError("Some required packages are not installed. Please restart WebUI to install them automatically.")

        parsed_prompt, parsed_extra_networks = parse_prompt(target_prompt)

        if target_image is None and parsed_prompt is None:
            raise ValueError("Nothing to process")

        device_name = available_devices[device_name_index][0] if ALLOW_DEVICE_SELECTION else this.model_device_name

        shared.state.textinfo = "Loading model..."
        model, preprocess = load_model(model_index, device_name)

        args.prompt_len = min(int(prompt_length), 75) if prompt_length is not None else 8
        args.iter = int(iterations_count) if iterations_count is not None else 1000

        shared.state.textinfo = "Processing..."

        prompt = utils.optimize_prompt(
            model,
            preprocess,
            args,
            torch.device(device_name),
            target_images = [ target_image ] if not target_image is None else None,
            target_prompts = [ parsed_prompt ] if not parsed_prompt is None else None,
            on_progress = on_progress,
            progress_steps = [ max(args.iter // 100, 10) ]
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

def inference_image(task_id, target_image, model_index, device_name_index, prompt_length, iterations_count):
    this.start_progress("Processing image")
    return inference(task_id, model_index, device_name_index, prompt_length, iterations_count, target_image = target_image)

def inference_text(task_id, target_prompt, model_index, device_name_index, prompt_length, iterations_count):
    this.start_progress("Processing prompt")
    return inference(task_id, model_index, device_name_index, prompt_length, iterations_count, target_prompt = target_prompt if not target_prompt is None and target_prompt != "" else None)

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
    with gr.Blocks(analytics_enabled = False) as tab:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Tab("Image to prompt"):
                        input_image = gr.Image(type = "pil", label = "Target image", show_label = False, elem_id = "pezdispenser_input_image")
                        process_image_button = gr.Button("Generate prompt", variant = "primary", elem_id = "pezdispenser_process_image_button")

                    with gr.Tab("Long prompt to short prompt"):
                        input_text = gr.TextArea(label = "Target prompt", show_label = False, interactive = True, elem_id = "pezdispenser_input_text")
                        process_text_button = gr.Button("Distill prompt", variant = "primary", elem_id = "pezdispenser_process_text_button")

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            opt_model = gr.Dropdown(label = "Model", choices = [n for n, _, _ in pretrained_models], type = "index",
                                value = pretrained_models[this.model_index][0], elem_id = "pezdispenser_opt_model")
                            unload_model_button = ToolButton(unload_symbol, elem_id = "pezdispenser_unload_model_button")

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

        unload_model_button.click(
            unload_model
        )

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

    for obj in [
        input_image,
        input_text,
        opt_model,
        opt_device,
        opt_prompt_length,
        opt_num_step
    ]:
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

        gr.HTML('<br />')
        with gr.Row():
            input_type = gr.Radio(label = 'Source', show_label = False,
                choices = [ "Long prompt to short prompt", "Image to prompt" ], value = "Long prompt to short prompt", elem_id = "pezdispenser_script_input_type")
        with gr.Row():
            input_image = gr.Image(type = "pil", label = "Target image", show_label = False, visible = False, elem_id = "pezdispenser_script_input_image")

        gr.HTML('<br />')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    opt_model = gr.Dropdown(label = "Model", choices = [n for n, _, _ in pretrained_models], type = "index",
                        value = pretrained_models[this.model_index][0], elem_id = "pezdispenser_script_opt_model")
                    unload_model_button = ToolButton(unload_symbol, elem_id = "pezdispenser_script_unload_model_button")

        gr.HTML('<br />')
        with gr.Row():
            with gr.Column():
                opt_prompt_length = gr.Slider(label = "Prompt length (optimal 8-16)", minimum = 1, maximum = 75, step = 1, value = args.prompt_len, elem_id = "pezdispenser_script_opt_prompt_length")
            with gr.Column():
                opt_num_step = gr.Slider(label = "Optimization steps (optimal 1000-3000)", minimum = 1, maximum = 10000, step = 1, value = args.iter, elem_id = "pezdispenser_script_opt_num_step")
            with gr.Column():
                opt_sample_step = gr.Slider(label = "Sample on every step (0 - disabled)", minimum = 0, maximum = 10000, step = 1, value = 0, elem_id = "pezdispenser_script_opt_sample_step")

        input_type.change(
            fn = None,
            _js = "pezdispenser_show_script_image",
            inputs = [input_type]
        )

        unload_model_button.click(
            unload_model
        )

        res = [
            input_type,
            input_image,
            opt_model,
            opt_prompt_length,
            opt_num_step,
            opt_sample_step
        ]

        for obj in res:
            setattr(obj, "do_not_save_to_config", True)

        return res

    def run(self, p,
        input_type,
        input_image,
        model_index,
        prompt_length,
        iterations_count,
        sample_every_iteration
    ):
        if not show_script():
            return Processed(p, [])

        script_name = os.path.splitext(os.path.basename(self.filename))[0]

        if not utils.state.installed:
            raise ModuleNotFoundError("Some required packages are not installed")

        is_image = input_type == "Image to prompt"
        parsed_prompt, parsed_extra_networks = parse_prompt(None if is_image else p.prompt)

        if is_image:
            if input_image is None:
                raise ValueError("Input image is empty")
        else:
            if parsed_prompt is None:
                raise RuntimeError("Prompt is empty")
                
        iterations_count_norm = int(iterations_count) if iterations_count is not None else 1000

        shared.state.job_count = p.n_iter

        argsc = copy.deepcopy(args)
        argsc.prompt_len = min(int(prompt_length), 75) if prompt_length is not None else 8
        argsc.iter = iterations_count_norm

        saved_textinfo = shared.state.textinfo
        shared.state.textinfo = "Loading model..."
        device_name = devices.get_optimal_device_name()
        model, preprocess = load_model(model_index, device_name)
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

            optimized_prompt = utils.optimize_prompt(
                model,
                preprocess,
                argsc,
                torch.device(device_name),
                target_images = [ input_image ] if is_image else None,
                target_prompts = None if is_image else [ parsed_prompt ],
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
