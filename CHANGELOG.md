## 1.6.1

### Features:
 * Experimental: Model precision selection. fp32 - default, fp16/bf16 - use almost 2x less memory and increase processing speed up to 1.5x without without noticeable quality loss.


## 1.6.0

### Features:
 * Ability to split prompt by lines
 * Experimental: Optimizer selection
 * Experimental: Torch compilation level. 0 - disabled, 1 - default, 2 - max-autotune-no-cudagraphs, 3 - max-autotune. Requires Triton and external C++ compiler. May speed up processing by 2-3 times. May crash WebUI or cause large GPU memory usage.


## 1.5.0

### Features:
 * Ability to use prompt as addition to image and in batch processing


## 1.4.3

### Bug Fixes:
 * Removed incompatible models selection


## 1.4.2

### Bug Fixes:
 * Fixed an issue with additional images in batch processing


## 1.4.1

### Bug Fixes:
 * Fixed incompatibility issue with WebUI 1.8.0


## 1.4.0

### Features:
 * Extra images for batch processing


## 1.3.2

### Features:
 * New option in script section: Do not make repetitive samples


## 1.3.1

### Features:
 * Unload model notification

### Bug Fixes:
 * Minor batch images processing fixes


## 1.3.0

### Features:
 * Batch images processing in script section


## 1.2.5

### Features:
 * Predefined SDXL model
 * Filtering special characters in result


## 1.2.4

### Features:
 * Unload model when UI reloads

### Bug Fixes:
 * Minor UI fixes


## 1.2.3

### Features:
 * Added link to Hard Prompts Made Easy documentation

### Bug Fixes:
 * Minor UI fixes


## 1.2.2

### Features:
 * Unload model button tooltip


## 1.2.1

### Bug Fixes:
 * Unload model small fix


## 1.2.0

### Features:
 * Multiple images support


## 1.1.0

### Features:
 * Advanced options in UI

### Bug Fixes:
 * Fix tokenizer selection


## 1.0.1

### Features:
 * Unload model button


## 1.0.0
 * Initial release
