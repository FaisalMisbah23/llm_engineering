import time

import gradio as gr

from analytics import (
    export_json,
    log_interaction,
    plot_daily_usage,
    plot_provider_avg_latency,
    summarize,
)
from audio import transcribe_audio
from code_generator import generate_cpp
from compiler import compile_cpp
from factory import ProviderFactory
from models import ModelManager


mm = ModelManager()


def _provider_models(provider: str) -> gr.Dropdown:
    models = mm.get_models(provider)
    return gr.Dropdown(choices=models, value=(models[0] if models else None))


def ask(provider, model, expertise, history, question):
    client = ProviderFactory.create(provider)
    system = mm.system_prompt(expertise)

    # Gradio `Chatbot(type="messages")` can include extra keys like `metadata`.
    # OpenAI-compatible APIs reject unknown fields, so strip to role/content only.
    clean_history = [
        {"role": m.get("role"), "content": m.get("content")}
        for m in (history or [])
        if isinstance(m, dict) and m.get("role") and m.get("content") is not None
    ]
    messages = [{"role": "system", "content": system}] + clean_history + [{"role": "user", "content": question}]

    t0 = time.time()
    ok, err = True, None
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        ok, err = False, str(e)
        answer = f"Error: {e}"

    dt = time.time() - t0
    log_interaction(provider, model, expertise, dt, ok, err)

    new_history = clean_history + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return new_history, ""


def do_transcribe(stt_provider, audio_path, stt_model):
    client = ProviderFactory.create(stt_provider)
    try:
        return transcribe_audio(client=client, audio_path=audio_path, model=stt_model)
    except Exception as e:
        return f"Transcription error: {e}"


def refresh_analytics():
    return summarize(), plot_provider_avg_latency(), plot_daily_usage()


def do_codegen(provider, model, expertise, py_or_prompt):
    cpp = generate_cpp(provider, model, expertise, py_or_prompt)
    comp = compile_cpp(cpp)
    status = "Compiled OK" if comp["ok"] else f"Compile failed:\n{comp['stderr']}"
    return cpp, status


def launch_ui():
    with gr.Blocks(title="Questher v3") as demo:
        gr.Markdown("## Questher v3")

        with gr.Row():
            provider = gr.Dropdown(choices=mm.list_providers(), value=mm.list_providers()[0], label="Provider")
            model = gr.Dropdown(
                choices=mm.get_models(mm.list_providers()[0]),
                value=mm.get_models(mm.list_providers()[0])[0],
                label="Model",
            )
            expertise = gr.Dropdown(choices=mm.list_expertise(), value="General", label="Expertise")

        provider.change(fn=_provider_models, inputs=[provider], outputs=[model])

        with gr.Tabs():
            with gr.Tab("Q&A"):
                chatbot = gr.Chatbot(type="messages", height=450, label="Conversation")
                question = gr.Textbox(label="Ask a question", placeholder="Type your technical question…")

                with gr.Row():
                    ask_btn = gr.Button("Ask")
                    clear_btn = gr.Button("Clear")

                ask_btn.click(
                    fn=ask,
                    inputs=[provider, model, expertise, chatbot, question],
                    outputs=[chatbot, question],
                )
                clear_btn.click(fn=lambda: [], inputs=[], outputs=[chatbot])

                gr.Markdown("### Speech-to-Text")
                audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload audio")
                transcribe_btn = gr.Button("Transcribe to question")
                transcribe_btn.click(fn=lambda audio_path: do_transcribe("GROQ", audio_path, "whisper-large-v3-turbo"), inputs=[audio], outputs=[question])

            with gr.Tab("Code Generation"):
                py_in = gr.Textbox(lines=10, label="Python code or prompt")
                gen_btn = gr.Button("Generate C++ + compile check")
                cpp_out = gr.Textbox(lines=14, label="Generated C++")
                compile_status = gr.Textbox(lines=6, label="Compile status")
                gen_btn.click(fn=do_codegen, inputs=[provider, model, expertise, py_in], outputs=[cpp_out, compile_status])

            with gr.Tab("Analytics"):
                stats_json = gr.JSON(label="Summary")
                fig_latency = gr.Plot(label="Provider latency")
                fig_daily = gr.Plot(label="Daily usage")

                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    export_btn = gr.Button("Export logs (JSON)")
                export_box = gr.Textbox(lines=10, label="Exported JSON")

                refresh_btn.click(fn=refresh_analytics, inputs=[], outputs=[stats_json, fig_latency, fig_daily])
                export_btn.click(fn=lambda: export_json(), inputs=[], outputs=[export_box])

    demo.launch()


if __name__ == "__main__":
    launch_ui()