# Riffusion extension for AUTOMATIC1111 Web UI

![Screenshot](/static/screenshot-0.png)

![Screenshot](/static/screenshot-1.png)

## Installation
- Make sure `ffmpeg` is installed and the folder with the binaries is in your `PATH`
- Clone this repo inside your `/extensions` folder, or use the Install from URL functionality in the UI

## Usage

Select the `Riffusion Audio Generator` script before generating, and use the riffusion model.

## Prompt Travelling

If you want to prompt travel in the latent space as described by the authors, install this extension:

https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel

It will output the results of runs in the `<SD>/outputs/(txt|img)2img-images/prompt_travel/` directory, and you can use the convert folder to audio functionality in the `Riffusion` tab to generate a single stitched-together audio file alongside the individual ones.

Here is a sample made by travelling in img2img mode from `jamaican rap` to `deep house, techno` with denoise `0.5` for `14` steps, and using the `og_beat.png` provided by the original authors as a base image:

[Audio Sample (Jamaican Rap to Deep House, Techno)](https://soundcloud.com/enlyth/sd-riffusion-prompt-travel-sample-0)

## Acknowledgements

Credit to the original Riffusion authors, Seth Forsgren and Hayk Martiros:

https://riffusion.com/about
