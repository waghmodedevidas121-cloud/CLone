import gradio as gr
import torch
import os
from TTS.api import TTS

# 1. Check for GPU (CUDA) availability and use it for faster inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device selected: {device}")

# 2. Load the XTTS-v2 model
# This is a high-quality, realistic zero-shot voice cloning model by Coqui
print("Loading XTTS-v2 model...")
# Setting COQUI_TOS_AGREED=1 to agree to terms of service automatically for XTTS-v2
os.environ["COQUI_TOS_AGREED"] = "1"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("Model loaded successfully!")

def clone_voice(text, reference_audio, language):
    """
    Function to generate cloned voice based on reference audio and text.
    """
    # Basic error handling
    if not text or text.strip() == "":
        raise gr.Error("Please provide text to synthesize.")
    if not reference_audio:
        raise gr.Error("Please upload a reference audio file.")
    if not language:
        raise gr.Error("Please select a language.")

    # Map language names to language codes required by XTTS-v2
    lang_map = {
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Polish": "pl",
        "Turkish": "tr",
        "Russian": "ru",
        "Dutch": "nl",
        "Czech": "cs",
        "Arabic": "ar",
        "Chinese": "zh-cn",
        "Hungarian": "hu",
        "Korean": "ko",
        "Japanese": "ja"
    }

    lang_code = lang_map.get(language, "en")
    output_path = "cloned_output.wav"

    try:
        # Generate voice using the TTS model
        tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language=lang_code,
            file_path=output_path
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"An error occurred during synthesis: {str(e)}")

# 3. Build Web Interface (UI) using Gradio
with gr.Blocks(title="Zero-Shot Voice Cloning") as demo:
    gr.Markdown("# Zero-Shot Voice Cloning with XTTS-v2")
    gr.Markdown("Clones a voice from a reference audio file to speak custom text in multiple languages. Optimized for GPU.")

    with gr.Row():
        with gr.Column():
            # a) Upload reference audio (.wav or .mp3, 30s to 2m long)
            ref_audio = gr.Audio(label="Reference Audio (30s to 2 mins recommended)", type="filepath")

            # b) Type or paste custom text
            text_input = gr.Textbox(label="Text to Synthesize", lines=5, placeholder="Type or paste the text you want the voice to speak...")

            # c) Select a language
            lang_dropdown = gr.Dropdown(
                choices=["English", "Hindi", "Spanish", "French", "German", "Italian", "Portuguese",
                         "Polish", "Turkish", "Russian", "Dutch", "Czech", "Arabic", "Chinese",
                         "Hungarian", "Korean", "Japanese"],
                value="English",
                label="Language"
            )

            # d) Generate button
            generate_btn = gr.Button("Generate Voice", variant="primary")

        with gr.Column():
            # Synthesize and play/download output audio
            output_audio = gr.Audio(label="Generated Output Audio", type="filepath", interactive=False)

    # Connect UI components to the voice cloning function
    generate_btn.click(
        fn=clone_voice,
        inputs=[text_input, ref_audio, lang_dropdown],
        outputs=output_audio
    )

if __name__ == "__main__":
    # Launch the Gradio app
    # Check for GRADIO_SHARE environment variable to enable sharing
    share = os.environ.get('GRADIO_SHARE') == '1'
    demo.launch(share=share)
