r"""A small web page to try the model: type a Bangla sentence, see the answer.

Run:  python app.py                              (uses the adapter folder gemma_cpu_lora next to this file)
      python app.py --adapter D:\path\to\gemma_cpu_lora
      python app.py --lan                        (other computers on the same Wi-Fi can open it too)
      python app.py --fake                       (no model: stand-in answers, only to look at the page)
Then open the address it prints (http://127.0.0.1:7860) in your browser.
"""
import argparse, os
import gradio as gr
from homograph_lookup import WORDS_CSV, TAG, load_words, fix_sentence, get_models

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="gemma_cpu_lora", help="folder made by train_cpu.py")
ap.add_argument("--words", default=str(WORDS_CSV), help="CSV with a 'homograph' column")
ap.add_argument("--fake", action="store_true", help="stand-in instead of Gemma (page check only)")
ap.add_argument("--lan", action="store_true", help="let other computers on the same Wi-Fi/network open the page")
ap.add_argument("--host", action="store_true", help="listen on all addresses (used by Hugging Face Spaces)")
ap.add_argument("--port", type=int, default=7860)
a = ap.parse_args()
a.host = a.host or bool(os.environ.get("SPACE_ID"))   # on a Hugging Face Space, listen on all addresses

words, said = load_words(a.words)
generate, chat = get_models(a.adapter, a.fake, said)

EXAMPLES = [
    "আমি কেন যাব?",
    "আজ গল্পের আসর বেশ জমে উঠেছে।",
    "কাঠমিস্ত্রি করাত দিয়ে কাঠ কাটছেন।",
    "আমি প্রতিদিন সকালে এক কাপ চা খাই।",
    "What is the capital of Bangladesh?",
]


def run(sentence):
    if not sentence.strip():
        return "", "", "type a sentence first"
    r = fix_sentence(sentence, words, generate, chat)
    if r["called_model"]:
        status = "homograph found: " + ", ".join(r["found"]) + "  → model was called"
    else:
        status = "no homograph from the list  → adapter switched off, plain Gemma answered like a normal chat"
    return r["tagged"], r["output"], status


with gr.Blocks(title="Bangla homograph fixer") as demo:
    gr.Markdown(f"# Bangla homograph fixer\nWords in the list: {', '.join(sorted(words))}\n\n"
                f"The page puts `{TAG}` in front of each listed word, then asks the model. "
                "If no listed word is found, the adapter is switched off and plain Gemma 3 270M answers normally (it is tiny, so answers can be weak)." + ("\n\n**STAND-IN MODE: not Gemma.**" if a.fake else ""))
    box = gr.Textbox(label="Sentence", lines=2)
    btn = gr.Button("Run", variant="primary")
    tagged = gr.Textbox(label="Input sent to the model", interactive=False)
    out = gr.Textbox(label="Output", interactive=False)
    status = gr.Textbox(label="What happened", interactive=False)
    gr.Examples(EXAMPLES, inputs=box)
    btn.click(run, box, [tagged, out, status])
    box.submit(run, box, [tagged, out, status])

if a.lan:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))       # no data is sent; this only finds this PC's network address
        print(f"\nOn the same Wi-Fi, open:  http://{s.getsockname()[0]}:{a.port}\n")
    except OSError:
        print("\nRun ipconfig, find 'IPv4 Address', and open http://<that address>:%d on the other computer\n" % a.port)
    finally:
        s.close()
demo.launch(server_name="0.0.0.0" if (a.lan or a.host) else "127.0.0.1", server_port=a.port)
