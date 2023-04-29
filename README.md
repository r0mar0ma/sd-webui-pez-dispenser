# PEZ Dispenser Extension for Stable-Diffusion-WebUI

This is a port of [Hard Prompts Made Easy](https://github.com/YuxinWenRick/hard-prompts-made-easy) / [PEZ Dispenser](https://huggingface.co/spaces/tomg-group-umd/pez-dispenser).

![](screenshot.jpg)

## Configuration

Default processing paramaters can be changed by placing `config.json` file to the extension directory.

Parameters are the same as on [Hard Prompts Made Easy](https://github.com/YuxinWenRick/hard-prompts-made-easy) page.

Additional parameters:

- `device`: device that will be selected by default. May be `cpu`, `cuda`, `cuda:0`, etc. By default will use the same device as in WebUI.
