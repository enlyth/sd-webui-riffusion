# All the audio related code was shamelessly copied from the original author(s):
# https://github.com/hmartiro/riffusion-inference

import io
import typing as T
import os
import numpy as np
from PIL import Image
from scipy.io import wavfile
import torch
import torchaudio
import gradio as gr
from modules import scripts, script_callbacks
from modules.images import FilenameGenerator
from modules.processing import process_images
import os
import modules.shared as shared
from pedalboard.io import AudioFile
import glob
from datetime import datetime
import wave

base_dir = scripts.basedir()

MAX_BATCH_SIZE = 8


class RiffusionScript(scripts.Script):
    last_generated_files = []
    last_generated_labels = []

    def title(self):
        return "Riffusion Audio Generator"

    def process_wav(self, wav_file, preserve_wav=False):
        with AudioFile(wav_file) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate

        filename = wav_file.replace(".wav", ".mp3")
        RiffusionScript.last_generated_files.append(filename)

        with AudioFile(filename, "w", samplerate, audio.shape[0]) as f:
            f.write(audio)

        if not preserve_wav:
            os.remove(wav_file)

    def ui(self, is_img2img):
        path = os.path.join(base_dir, "outputs")

        with gr.Row():
            riffusion_enabled = gr.Checkbox(label="Riffusion enabled", value=True)
            save_wav = gr.Checkbox(label="Preserve Original WAV", value=False)

        output_path = gr.Textbox(label="Output path", value=path)

        def update_audio_players():
            count = len(RiffusionScript.last_generated_files)
            updates = [
                gr.Audio.update(
                    value=RiffusionScript.last_generated_files[i],
                    visible=True,
                    label=RiffusionScript.last_generated_labels[i],
                )
                for i in range(count)
            ]
            # pad with empty updates
            for _ in range(count, MAX_BATCH_SIZE):
                updates.append(gr.Audio.update(value=None, visible=False))
            return updates

        # create MAX_BATCH_SIZE audio players, and hide the unnecessary ones
        audio_players = []
        for i in range(MAX_BATCH_SIZE):
            audio_players.append(
                gr.Audio(
                    label=f"Audio Player {i}",
                    visible=False,
                    value=None,
                    interactive=False,
                )
            )

        show_audio_button = gr.Button(
            "Refresh Inline Audio (Last Batch)",
            label="Refresh Inline Audio (Last Batch)",
            variant="primary",
        )
        show_audio_button.click(
            fn=lambda: update_audio_players(),
            inputs=[],
            outputs=audio_players,
        )
        hide_audio_button = gr.Button("Hide Inline Audio", label="Hide Inline Audio")
        hide_audio_button.click(
            fn=lambda: [
                gr.Audio.update(value=None, visible=False)
                for _ in range(MAX_BATCH_SIZE)
            ],
            inputs=[],
            outputs=audio_players,
        )

        return [
            riffusion_enabled,
            save_wav,
            output_path,
            show_audio_button,
            *audio_players,
        ]

    def play_input_as_sound(self):
        pass

    def run(self, p, riffusion_enabled, save_wav, output_path, btn, *audio_players):
        if riffusion_enabled is False:
            return process_images(p)
        else:
            print("Generating Riffusion mp3")

        proc = process_images(p)

        RiffusionScript.last_generated_labels = []
        RiffusionScript.last_generated_files = []
        try:
            # try to create output path dir if doesnt exist
            os.makedirs(output_path)
        except FileExistsError:
            pass

        for i in range(len(proc.images)):
            wav_bytes, duration_s = self.wav_bytes_from_spectrogram_image(
                proc.images[i]
            )
            namegen = FilenameGenerator(p, p.seed, p.prompt, proc.images[i])
            name = namegen.apply(f"[job_timestamp]-[seed]-[prompt_spaces]-{i}")

            filename = os.path.join(output_path, f"{name}.wav")

            with open(filename, "wb") as f:
                f.write(wav_bytes.getbuffer())

            self.process_wav(filename, preserve_wav=save_wav)
            RiffusionScript.last_generated_labels.append(
                namegen.apply(f"[seed]-[prompt_spaces]-{i}")
            )

        return proc

    def wav_bytes_from_spectrogram_image(
        self,
        image: Image.Image,
    ) -> T.Tuple[io.BytesIO, float]:
        """
        Reconstruct a WAV audio clip from a spectrogram image. Also returns the duration in seconds.
        """

        max_volume = 50
        power_for_image = 0.25
        Sxx = self.spectrogram_from_image(
            image, max_volume=max_volume, power_for_image=power_for_image
        )

        sample_rate = 44100  # [Hz]
        clip_duration_ms = 5000  # [ms]

        bins_per_image = image.height
        n_mels = image.height

        # FFT parameters
        window_duration_ms = 100  # [ms]
        padded_duration_ms = 400  # [ms]
        step_size_ms = 10  # [ms]

        # Derived parameters
        num_samples = (
            int(image.width / float(bins_per_image) * clip_duration_ms) * sample_rate
        )
        n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
        hop_length = int(step_size_ms / 1000.0 * sample_rate)
        win_length = int(window_duration_ms / 1000.0 * sample_rate)

        samples = self.waveform_from_spectrogram(
            Sxx=Sxx,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            num_samples=num_samples,
            sample_rate=sample_rate,
            mel_scale=True,
            n_mels=n_mels,
            max_mel_iters=200,
            num_griffin_lim_iters=32,
        )

        wav_bytes = io.BytesIO()
        wavfile.write(wav_bytes, sample_rate, samples.astype(np.int16))
        wav_bytes.seek(0)

        duration_s = float(len(samples)) / sample_rate

        return wav_bytes, duration_s

    def spectrogram_from_image(
        self, image: Image.Image, max_volume: float = 50, power_for_image: float = 0.25
    ) -> np.ndarray:
        """
        Compute a spectrogram magnitude array from a spectrogram image.
        TODO(hayk): Add image_from_spectrogram and call this out as the reverse.
        """
        # Convert to a numpy array of floats
        data = np.array(image).astype(np.float32)

        # Flip Y take a single channel
        data = data[::-1, :, 0]

        # Invert
        data = 255 - data

        # Rescale to max volume
        data = data * max_volume / 255

        # Reverse the power curve
        data = np.power(data, 1 / power_for_image)

        return data

    def spectrogram_from_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        mel_scale: bool = True,
        n_mels: int = 512,
    ) -> np.ndarray:
        """
        Compute a spectrogram from a waveform.
        """

        spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            power=None,
            hop_length=hop_length,
            win_length=win_length,
        )

        waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).reshape(1, -1)
        Sxx_complex = spectrogram_func(waveform_tensor).numpy()[0]

        Sxx_mag = np.abs(Sxx_complex)

        if mel_scale:
            mel_scaler = torchaudio.transforms.MelScale(
                n_mels=n_mels,
                sample_rate=sample_rate,
                f_min=0,
                f_max=10000,
                n_stft=n_fft // 2 + 1,
                norm=None,
                mel_scale="htk",
            )

            Sxx_mag = mel_scaler(torch.from_numpy(Sxx_mag)).numpy()

        return Sxx_mag

    def waveform_from_spectrogram(
        self,
        Sxx: np.ndarray,
        n_fft: int,
        hop_length: int,
        win_length: int,
        num_samples: int,
        sample_rate: int,
        mel_scale: bool = True,
        n_mels: int = 512,
        max_mel_iters: int = 200,
        num_griffin_lim_iters: int = 32,
        device: str = "cuda:0",
    ) -> np.ndarray:
        """
        Reconstruct a waveform from a spectrogram.
        This is an approximate inverse of spectrogram_from_waveform, using the Griffin-Lim algorithm
        to approximate the phase.
        """
        Sxx_torch = torch.from_numpy(Sxx).to(device)

        # TODO(hayk): Make this a class that caches the two things

        if mel_scale:
            mel_inv_scaler = torchaudio.transforms.InverseMelScale(
                n_mels=n_mels,
                sample_rate=sample_rate,
                f_min=0,
                f_max=10000,
                n_stft=n_fft // 2 + 1,
                norm=None,
                mel_scale="htk",
                max_iter=max_mel_iters,
            ).to(device)

            Sxx_torch = mel_inv_scaler(Sxx_torch)

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
            n_iter=num_griffin_lim_iters,
        ).to(device)

        waveform = griffin_lim(Sxx_torch).cpu().numpy()

        return waveform


def convert_audio_file(image, output_dir):
    image_file = Image.open(image)
    new_filename = os.path.splitext(os.path.basename(image))[0] + ".wav"
    filename = os.path.join(output_dir, new_filename)
    riffusion = RiffusionScript()
    wav_bytes, duration_s = riffusion.wav_bytes_from_spectrogram_image(image_file)

    with open(filename, "wb") as f:
        f.write(wav_bytes.getbuffer())
    return filename


def convert_audio(image_dir: str, file_regex: str, join_images: bool) -> None:

    images = []

    globs = map(lambda x: x.strip(), file_regex.split(","))

    for g in globs:
        images.extend(glob.glob(os.path.join(image_dir, g)))

    print(f"Found {len(images)} images in {image_dir}, pattern {file_regex}")
    output_files = []
    for image in images:
        output_files.append(convert_audio_file(image, image_dir))

    if join_images and len(output_files) > 1:
        data = []
        outfile = os.path.join(
            image_dir,
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_joined.wav",
        )
        for wav in output_files:
            w = wave.open(wav, "rb")
            data.append([w.getparams(), w.readframes(w.getnframes())])
            w.close()
        output = wave.open(outfile, "wb")
        output.setparams(data[0][0])
        for i in range(len(data)):
            output.writeframes(data[i][1])
        output.close()

    print(f"Converted {len(images)} images to audio")


def on_ui_tabs():
    with gr.Blocks() as riffusion_ui:
        with gr.Row():
            with gr.Column(variant="panel"):
                with gr.Row():
                    image_directory = gr.Textbox(
                        label="Image Directory",
                        placeholder="Directory containing your image files",
                        value="",
                        interactive=True,
                    )
                with gr.Row():
                    join_images = gr.Checkbox(
                        label="Also output single joined audio file (will be named <date>_joined.wav)",
                        value=True,
                        interactive=True,
                    )
                with gr.Row():
                    file_regex = gr.Textbox(
                        label="GLOB patterns (comma separated)",
                        value="*.jpg, *.png",
                        interactive=True,
                    )
            with gr.Column(variant="panel"):
                with gr.Row():
                    convert_folder_btn = gr.Button(
                        "Convert Folder", label="Convert Folder", variant="primary"
                    )
                    convert_folder_btn.click(
                        convert_audio,
                        inputs=[
                            image_directory,
                            file_regex,
                            join_images,
                        ],
                        outputs=[],
                    )
                gr.HTML(value="<p>Converts all images in a folder to audio</p>")
    return ((riffusion_ui, "Riffusion", "riffusion_ui"),)


script_callbacks.on_ui_tabs(on_ui_tabs)
