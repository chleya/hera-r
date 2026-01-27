import gradio as gr
import torch
import yaml
from hera.core.engine import HeraEngine

engine = None

def init_system():
    global engine
    try:
        if engine is None:
            yield "⏳ Initializing (loading Model & SAE)..."
            engine = HeraEngine("configs/default.yaml")
            yield f"✅ Ready!\nDevice: {engine.cfg['experiment']['device']}"
        else:
            yield "✅ Already running."
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def run_evolution(prompt, lr, threshold):
    global engine
    if engine is None:
        return "System not initialized", {}, "N/A"

    engine.cfg["evolution"]["learning_rate"] = lr
    engine.cfg["sae"]["threshold"] = threshold
    
    with torch.no_grad():
        pre_gen = engine.model.generate(prompt, max_new_tokens=15, verbose=False)
    
    try:
        success = engine.evolve(prompt)
        status = "✅ Committed" if success else "🛡️ Rejected"
    except Exception as e:
        return f"Error: {str(e)}", {}, "Error"

    with torch.no_grad():
        post_gen = engine.model.generate(prompt, max_new_tokens=15, verbose=False)

    metrics = engine.registry.history[-1]["metrics"] if engine.registry.history else {}
    return f"### Before:\n{pre_gen}\n\n### After:\n{post_gen}", metrics, status

with gr.Blocks(title="H.E.R.A.-R", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 H.E.R.A.-R Control Panel")
    with gr.Row():
        with gr.Column():
            init_btn = gr.Button("🚀 Initialize System", variant="primary")
            sys_status = gr.Textbox(label="Status")
            lr_slider = gr.Slider(0.001, 0.1, value=0.01, label="Learning Rate")
            thresh_slider = gr.Slider(0.5, 5.0, value=2.0, label="SAE Threshold")
        with gr.Column():
            input_text = gr.Textbox(label="Stimulus Prompt", value="The capital of France is")
            run_btn = gr.Button("🧬 Evolve", variant="secondary")
            output_display = gr.Markdown(label="Output")
            metrics_json = gr.JSON(label="Immune Metrics")
            status_box = gr.Label(label="Result")

    init_btn.click(init_system, outputs=[sys_status])
    run_btn.click(run_evolution, inputs=[input_text, lr_slider, thresh_slider], outputs=[output_display, metrics_json, status_box])

if __name__ == "__main__":
    demo.queue().launch()