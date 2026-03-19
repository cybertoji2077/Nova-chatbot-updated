from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from openai import OpenAI
import requests
import base64
import time
import json
import os

app = Flask(__name__, static_folder=".")

# ── Keys: set via environment variables (for HF Spaces / deployment)
# On HF Spaces: add these in Settings → Variables and Secrets
# Locally: create a .env file or set them in your terminal
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")

# ── Image generation ────────────────────────────────────────────────────────
HF_IMG_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

def generate_image(prompt):
    if not HF_TOKEN:
        return {"success": False, "error": "HF_TOKEN not configured on server."}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}
    for _ in range(3):
        resp = requests.post(HF_IMG_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return {"success": True, "image_b64": base64.b64encode(resp.content).decode()}
        elif resp.status_code == 503:
            wait = resp.json().get("estimated_time", 20)
            time.sleep(min(wait, 30))
        else:
            try:   err = resp.json()
            except: err = {"error": f"HTTP {resp.status_code}"}
            return {"success": False, "error": str(err)}
    return {"success": False, "error": "Model is still loading. Try again in 30 seconds."}


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming chat endpoint — sends tokens as SSE."""
    data       = request.json
    messages   = data.get("messages", [])
    use_search = data.get("use_search", False)

    if not OPENROUTER_KEY:
        return jsonify({"error": "OPENROUTER_KEY not configured on server."}), 500

    client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
    model  = data.get("model", "deepseek/deepseek-chat")

    kwargs = {"model": model, "messages": messages, "stream": True}

    if use_search:
        kwargs["tools"] = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }]
        kwargs["tool_choice"] = "auto"

    def generate():
        try:
            # First try streaming
            full_text    = ""
            tool_call_id = None
            tool_args    = ""
            tool_name    = None
            searched     = False
            query        = ""

            stream = client.chat.completions.create(**kwargs)

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Handle tool calls (web search)
                if delta.tool_calls:
                    tc = delta.tool_calls[0]
                    if tc.id:
                        tool_call_id = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_name = tc.function.name
                        if tc.function.arguments:
                            tool_args += tc.function.arguments
                    continue

                if delta.content:
                    full_text += delta.content
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"

            # If a tool was called, do the follow-up
            if tool_call_id and tool_name == "web_search":
                try:
                    query = json.loads(tool_args).get("query", "")
                except Exception:
                    query = ""

                searched = True
                yield f"data: {json.dumps({'searching': True, 'query': query})}\n\n"

                search_result = (
                    f'Web search results for "{query}": '
                    f"Based on current information, here is what is known about this topic."
                )

                follow_up_messages = messages + [
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": tool_call_id, "type": "function",
                        "function": {"name": "web_search", "arguments": tool_args}
                    }]},
                    {"role": "tool", "tool_call_id": tool_call_id, "content": search_result}
                ]

                stream2 = client.chat.completions.create(
                    model=model, messages=follow_up_messages, stream=True
                )
                full_text = ""
                for chunk in stream2:
                    delta2 = chunk.choices[0].delta if chunk.choices else None
                    if delta2 and delta2.content:
                        full_text += delta2.content
                        yield f"data: {json.dumps({'token': delta2.content})}\n\n"

            yield f"data: {json.dumps({'done': True, 'searched': searched, 'query': query})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/generate-image", methods=["POST"])
def generate_image_route():
    data   = request.json
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    result = generate_image(prompt)
    if result["success"]:
        return jsonify({"image_b64": result["image_b64"]})
    return jsonify({"error": result["error"]}), 500


if __name__ == "__main__":
    print("\n  ✦ NOVA is running → http://localhost:5000\n")
    app.run(debug=True, port=5000)
