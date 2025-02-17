import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from vocoder import inference as vocoder


if _name_ == '_main_':
    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="/content/drive/MyDrive/saved_models/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="/content/drive/MyDrive/saved_models/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="/content/drive/MyDrive/saved_models/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help="Disable audio playback.")
    parser.add_argument("--seed", type=int, default=None, help="Set a seed for deterministic results.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Force CPU if specified
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    # Check GPU or CPU
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print(f"Found {torch.cuda.device_count()} GPUs available. "
              f"Using GPU {device_id} ({gpu_properties.name}) with "
              f"{gpu_properties.total_memory / 1e9:.1f}GB total memory.\n")
    else:
        print("Using CPU for inference.\n")

    # Load models
    print("Preparing the encoder, synthesizer, and vocoder...")

    # Verify and load encoder model
    encoder_model_path = args.enc_model_fpath
    if not encoder_model_path.exists():
        raise FileNotFoundError(f"Encoder model file not found at {encoder_model_path}")
    print(f"Loading encoder model from {encoder_model_path}...")
    encoder.load_model(encoder_model_path)
    print("Encoder model loaded successfully.")

    # Verify and load synthesizer model
    synthesizer_model_path = args.syn_model_fpath
    if not synthesizer_model_path.exists():
        raise FileNotFoundError(f"Synthesizer model file not found at {synthesizer_model_path}")
    print(f"Loading synthesizer model from {synthesizer_model_path}...")
    synthesizer = Synthesizer(synthesizer_model_path)
    print("Synthesizer model loaded successfully.")

    # Verify and load vocoder model
    vocoder_model_path = args.voc_model_fpath
    if not vocoder_model_path.exists():
        raise FileNotFoundError(f"Vocoder model file not found at {vocoder_model_path}")
    print(f"Loading vocoder model from {vocoder_model_path}...")
    vocoder.load_model(vocoder_model_path)
    print("Vocoder model loaded successfully.")

    # Test encoder
    print("Testing encoder with dummy input...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    print("Encoder test passed.")

    # Interactive speech generation loop
    print("Interactive generation loop")
    num_generated = 0
    while True:
        try:
            # Get the reference audio filepath
            print("\nProvide the path to your reference audio file (e.g., .wav, .mp3):")
            in_fpath = input("Reference audio file path: ").strip()
            in_fpath = Path(in_fpath)

            if not in_fpath.exists():
                print(f"File {in_fpath} does not exist. Please provide a valid file path.")
                continue

            # Preprocess the reference audio file
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            print("Loaded and preprocessed the reference audio successfully.")

            # Generate the embedding
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Generated speaker embedding.")

            # Input text for synthesis
            text = input("Enter the text you want to synthesize (20 words max):\n").strip()
            if not text:
                print("No text provided. Please enter some text.")
                continue

            # Generate mel spectrogram
            texts = [text]
            embeds = [embed]
            print("Generating mel spectrogram...")
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Mel spectrogram created.")

            # Generate waveform using vocoder
            print("Synthesizing waveform...")
            generated_wav = vocoder.infer_waveform(spec)

            # Pad and trim the generated waveform
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            generated_wav = encoder.preprocess_wav(generated_wav)

            # Save the output to Google Drive
            output_path = f"/content/drive/MyDrive/demo_output_{num_generated:02d}.wav"
            sf.write(output_path, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print(f"Audio saved to: {output_path}")

        except Exception as e:
            print(f"Error occurred: {repr(e)}. Restarting loop.")