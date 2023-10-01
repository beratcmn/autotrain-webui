import gradio as gr
import os


def train():
    pass


with gr.Blocks() as demo:
    gr.Markdown("# AutoTrain Advanced WebUI")

    gr.Markdown("## Project Config")
    with gr.Group():
        with gr.Row():
            project_name = gr.Textbox("", label="Project Name",
                                      placeholder="Enter a name for the project", interactive=True)
            model_name = gr.Textbox("", label="Model Name",
                                    placeholder="Base Model ex. `meta-llama/Llama-2-7b`", interactive=True)

    with gr.Group():
        with gr.Column():
            push_to_hub = gr.Checkbox(value=False,
                                      label="Push to Huggingface Hub", interactive=True)

            hf_token = gr.Textbox("", label="Huggingface Token",
                                  placeholder="Enter your Huggingface token", interactive=True)

            hf_repo = gr.Textbox("", label="Huggingface Repo",
                                 placeholder="Enter your Huggingface repo", interactive=True)

    with gr.Group():
        with gr.Column():
            learning_rate = gr.Textbox("2e-4", label="Learning Rate",
                                       placeholder="Enter a learning rate", interactive=True)

            num_epochs = gr.Textbox("1", label="Number of Epochs",
                                    placeholder="Enter a number of epochs", interactive=True)

            batch_size = gr.Slider(1, 32, 1, label="Batch Size",
                                   description="Enter a batch size", interactive=True)

            block_size = gr.Textbox("1024", label="Block Size",
                                    placeholder="Enter a block size", interactive=True)

            trainer_args = gr.Textbox("", label="Trainer Args",
                                      placeholder="Enter a trainer, 'sft' or 'default'", interactive=True)

            warmup_ratio = gr.Textbox("0.1", label="Warmup Ratio",
                                      placeholder="Enter a warmup ratio", interactive=True)

            weight_decay = gr.Textbox("0.01", label="Weight Decay",
                                      placeholder="Enter a weight decay", interactive=True)

            gradient_accumulation = gr.Textbox("4", label="Gradient Accumulation",
                                               placeholder="Enter a gradient accumulation", interactive=True)

            use_fp16 = gr.Checkbox(value=True, label="Use FP16", interactive=True)

            use_peft = gr.Checkbox(value=True, label="Use PEFT", interactive=True)

            use_int4 = gr.Checkbox(value=True, label="Use INT4", interactive=True)

            lora_r = gr.Textbox("16", label="LoRA R",
                                placeholder="Enter a LoRA R", interactive=True)

            lora_alpha = gr.Textbox("32", label="LoRA Alpha",
                                    placeholder="Enter a LoRA Alpha", interactive=True)

            lora_dropout = gr.Textbox("0.05", label="LoRA Dropout",
                                      placeholder="Enter a LoRA Dropout", interactive=True)

    train_button = gr.Button("Start Training")
    train_button.click(train)


def train():
    print("Starting training...")
    train_button.update(value="Training...", interactive=False)

    # Create the command
    os.system(
        "autotrain", "llm", "--train", f"--model {model_name.value}", f"--project-name {project_name.value}",
        "--data-path data/", "--text-column text", f"--lr {learning_rate.value}", f"--batch-size {batch_size.value}",
        f"--epochs {num_epochs.value}", f"--block-size {block_size.value}", f"--warmup-ratio {warmup_ratio.value}",
        f"--lora-r {lora_r.value}", f"--lora-alpha {lora_alpha.value}", f"--lora-dropout {lora_dropout.value}",
        f"--weight-decay {weight_decay.value}", f"--gradient-accumulation {gradient_accumulation.value}")


if __name__ == "__main__":
    demo.launch()
