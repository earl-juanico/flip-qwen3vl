#!/usr/bin/env python3
"""
query_indout.py

Reads an input JSONL of questions and queries the local chat-completions endpoint
with prompt "Is this photo taken indoors or outdoors? Answer one word." + image_url.
Writes one JSONL output line per input with keys:
    {"question_id": ..., "prompt": ..., "text": <response content>, "flip_vartheta": ...}

Supports runtime FLIP vartheta updates via --config-file: the patched server
(patch_qwen.py) picks up FLIP_VARTHETA changes without restart by polling the file mtime.
"""
import os
import re
import json
import argparse
import requests
import time
from typing import Optional

_OBJECT_RE = re.compile(r'of\s+(?:all\s+instances\s+of\s+)?(.+?)\s+in\b', flags=re.IGNORECASE)

# FLIP file cache for change detection
_flip_file_cached_value: Optional[float] = None
_flip_file_cached_mtime: Optional[float] = None


def extract_object_from_text(qtext: str) -> Optional[str]:
    m = _OBJECT_RE.search(qtext)
    if not m:
        return None
    return m.group(1).strip()


def build_payload(
    prompt_text: str, container_image_url: str, max_tokens: int = 64,
    temperature: float = 0.7,
    greedy: bool = False,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 1.5,
    out_seq_length: int = 16384,
    ) -> dict:
    """
    Build payload mirroring the curl/jq flow used by query_annotate.sh.
    Include explicit generation limits and request non-streaming responses via "stream": False.
    """
    return {
        "model": "/workspace/model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": container_image_url}}
                ]
            }
        ],
        # generation controls
        "max_output_tokens": max_tokens,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # request non-streaming output so server returns an aggregated content blob
        "stream": False,
        "greedy": greedy,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "out_seq_length": out_seq_length,
    }


def query_model(session: requests.Session, api_url: str, api_key: str, payload: dict, timeout: int = 60) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    resp = session.post(api_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_server_models(session: requests.Session, api_url_base: str, api_key: str, timeout: int = 5) -> list:
    """Query the server's /v1/models endpoint and return a list of model ids.

    Returns empty list on error or if endpoint not available.
    """
    url = api_url_base.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()
        data = j.get("data")
        if isinstance(data, list):
            ids = [d.get("id") for d in data if isinstance(d, dict) and d.get("id")]
            return ids
        if isinstance(j, list):
            return [it.get("id") for it in j if isinstance(it, dict) and it.get("id")]
        return []
    except Exception:
        return []


def extract_response_content(resp_json: dict) -> str:
    """
    Robustly extract textual content from the model response.
    Handles common shapes:
      - choices[0].message.content as string
      - content as list of blocks/dicts (join textual fields)
      - content as dict with 'text'/'content' keys
    Falls back to JSON dump.
    """
    try:
        choices = resp_json.get("choices")
        if choices and len(choices) > 0:
            msg = choices[0].get("message") or choices[0].get("delta") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, (list, tuple)):
                parts = []
                for c in content:
                    if isinstance(c, str):
                        parts.append(c)
                    elif isinstance(c, dict):
                        for k in ("text", "content", "items"):
                            v = c.get(k)
                            if isinstance(v, str):
                                parts.append(v)
                                break
                        else:
                            for v in c.values():
                                if isinstance(v, str):
                                    parts.append(v)
                                    break
                joined = " ".join(p.strip() for p in parts if p and p.strip())
                if joined:
                    return joined.strip()
            if isinstance(content, dict):
                for k in ("text", "content", "items"):
                    v = content.get(k)
                    if isinstance(v, str):
                        return v.strip()
                    if isinstance(v, (list, tuple)):
                        return " ".join(x for x in v if isinstance(x, str)).strip()
            return json.dumps(content, ensure_ascii=False)
    except Exception:
        pass
    # fallback common jq path used in the bash script
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        pass
    return json.dumps(resp_json, ensure_ascii=False)


def _parse_flip_text(raw: str) -> Optional[float]:
    txt = raw.strip()
    if txt == "" or txt.lower() == "none":
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _read_flip_file_if_changed(path: str) -> Optional[Optional[float]]:
    """
    Stat the FLIP vartheta file and, if mtime differs from cached, read and parse it.
    On update print an INFO line and update module cache. Returns the new parsed
    value if changed, or None if unchanged, or None/None if file missing.
    """
    global _flip_file_cached_mtime, _flip_file_cached_value
    try:
        st = os.stat(path)
    except FileNotFoundError:
        # if file removed after being present, clear cache and report change
        if _flip_file_cached_mtime is not None:
            _flip_file_cached_mtime = None
            _flip_file_cached_value = None
            print(f"[INFO] FLIP vartheta file removed: {path}; FLIP_VARTHETA cleared")
            return None
        return None
    except Exception as e:
        # non-fatal; do nothing
        print(f"[WARN] unable to stat FLIP file {path}: {e}")
        return None

    if _flip_file_cached_mtime is None or st.st_mtime != _flip_file_cached_mtime:
        # file changed -> read & parse
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            parsed = _parse_flip_text(raw)
            _flip_file_cached_mtime = st.st_mtime
            prev = _flip_file_cached_value
            _flip_file_cached_value = parsed
            # print only if the parsed value differs from previous
            if parsed != prev:
                print(f"[INFO] FLIP vartheta updated from file {path}: {_flip_file_cached_value}")
            return parsed
        except Exception as e:
            print(f"[WARN] error reading/parsing FLIP file {path}: {e}")
            return None
    return None


def _write_flip_file_atomic(path: str, value: str) -> None:
    """
    Atomically write 'value' to path (creates parent dir if needed). Writing the
    file updates its mtime so patch_qwen._get_flip_vartheta() will reload the value.
    Accepts any string (e.g. "0.1" or "None"). Also updates local cache.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(value) + "\n")
    os.replace(tmp, path)
    # Refresh cache immediately after write and emit INFO if changed
    try:
        _read_flip_file_if_changed(path)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Query model for each question and save JSONL responses.")
    p.add_argument("--questions", "-q", required=True, help="Input questions JSONL file")
    p.add_argument("--image-dir", "-i", default="/data/coco", help="Directory where images live on the host")
    p.add_argument("--api-port", "-p", type=int, default=8001, help="Local API port (default 8001)")
    p.add_argument("--output", "-o", default="qwen_outputs.jsonl", help="Output JSONL file")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests (default 0)")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "foobar"), help="API key (env VLLM_API_KEY)")
    p.add_argument("--max-tokens", type=int, default=64, help="Max output tokens (default 64)")
    p.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default 0.0)")
    p.add_argument("--model-path", "-m", default="/workspace/model", help="Model path or identifier to include in payload 'model' key")
    p.add_argument("--model-id", default=None, help="Optional model identifier to send to the server; overrides --model-path when set")
    # FLIP runtime controls
    default_flip_file = os.environ.get("FLIP_VARTHETA_FILE", "/workspace/config/vartheta.txt")
    p.add_argument("--config-file", default=default_flip_file, help=f"FLIP vartheta file path (default: {default_flip_file})")
    p.add_argument("--flip-vartheta", default=None, help="If set, write this value to --config-file before each query (e.g. '0.1' or 'None')")
    default_media_path = os.environ.get("MEDIA_PATH", "")
    p.add_argument("--media-path", default=default_media_path,
                   help="Root directory served by the HTTP image server (MEDIA_PATH). "
                        "When set, image URLs are computed as relative paths from this root "
                        "rather than using only the basename. (env MEDIA_PATH)")
    args = p.parse_args()

    api_url = f"http://localhost:{args.api_port}/v1/chat/completions"
    session = requests.Session()

    # Try to detect server-registered model ids and pick a matching one.
    server_models = get_server_models(session, f"http://localhost:{args.api_port}", args.api_key)
    if server_models:
        candidate = None
        if args.model_id:
            candidate = args.model_id if args.model_id in server_models else None
        else:
            if args.model_path in server_models:
                candidate = args.model_path
            else:
                try:
                    base = os.path.basename(args.model_path)
                except Exception:
                    base = args.model_path
                if base in server_models:
                    candidate = base
                else:
                    for m in server_models:
                        if m and m in args.model_path:
                            candidate = m
                            break
        if candidate:
            if candidate != (args.model_id or args.model_path):
                print(f"[INFO] Using server model id '{candidate}' (matched from --model-path)")
            args.model_id = candidate

    if not os.path.isfile(args.questions):
        raise SystemExit(f"Questions JSONL not found: {args.questions}")

    out_f = open(args.output, "w", encoding="utf-8")
    with open(args.questions, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {lineno}: invalid json: {e}; skipping")
                continue

            question_id = item.get("question_id")
            image_filename = item.get("image")
            text_field = item.get("text", "")

            if image_filename is None:
                print(f"[WARN] line {lineno}: missing 'image' key; skipping")
                continue

            host_image_path = os.path.join(args.image_dir, image_filename)
            if not os.path.isfile(host_image_path):
                print(f"[WARN] line {lineno}: image not found at {host_image_path}; continuing (server may still access container path)")

            # Derive the local HTTP media port from the API port (mirror serve_with_patch.py logic)
            def _derive_http_port(p: int) -> int:
                s = str(p)
                if s and s[0] == "8":
                    try:
                        return int("9" + s[1:])
                    except Exception:
                        pass
                try:
                    return int(p) + 1000
                except Exception:
                    return 9000

            http_port = _derive_http_port(args.api_port)

            # Prefer serving the image via the server's local HTTP server so vLLM
            # fetches it without local-path validation issues.
            # If --media-path is set, compute the relative path from the HTTP server root
            # (matching the cwd used by serve_with_patch.py's http.server process).
            # Otherwise fall back to basename only.
            media_root = getattr(args, "media_path", "") or ""
            if media_root:
                try:
                    abs_image = os.path.realpath(host_image_path)
                    abs_root = os.path.realpath(media_root)
                    rel = os.path.relpath(abs_image, abs_root)
                    if rel.startswith(".."):
                        rel = os.path.basename(host_image_path)
                except Exception:
                    rel = os.path.basename(host_image_path)
                url_path = rel.replace(os.sep, "/")
            else:
                url_path = os.path.basename(host_image_path)
            container_image_url = f"http://127.0.0.1:{http_port}/{url_path}"
            print(f"[INFO] line {lineno}: question_id={question_id}, image_url='{container_image_url}'")

            # Check FLIP file for external updates (prints INFO if changed)
            if args.config_file:
                try:
                    _read_flip_file_if_changed(args.config_file)
                except Exception as e:
                    print(f"[WARN] failed checking FLIP file {args.config_file}: {e}")

            # If requested, atomically write FLIP vartheta file so the running
            # patched server can pick up the new vartheta without restart.
            if args.flip_vartheta is not None:
                try:
                    _write_flip_file_atomic(args.config_file, args.flip_vartheta)
                    print(f"[INFO] wrote FLIP vartheta '{args.flip_vartheta}' to {args.config_file}")
                except Exception as e:
                    print(f"[WARN] failed writing FLIP file {args.config_file}: {e}")

            obj = extract_object_from_text(text_field)
            if not obj:
                alt = None
                if " of " in text_field:
                    try:
                        alt = text_field.split(" of ", 1)[1].split(" in ", 1)[0].strip()
                    except Exception:
                        alt = None
                obj = alt or text_field.strip()
            # global scene
            prompt = f"Is this photo taken indoors or outdoors? Answer one word."
            print(f"[INFO] line {lineno}: question_id={question_id}, prompt='{prompt}', FLIP_VARTHETA={_flip_file_cached_value}")

            payload = build_payload(prompt, container_image_url,
                        max_tokens=args.max_tokens, temperature=args.temperature
                    )
            # Ensure the payload 'model' field matches what we detected/passed.
            model_to_send = args.model_id if getattr(args, "model_id", None) is not None else args.model_path
            try:
                payload["model"] = model_to_send
            except Exception:
                pass
            try:
                resp_json = query_model(session, api_url, args.api_key, payload, timeout=3000)
            except Exception as e:
                print(f"[ERROR] line {lineno}: model query failed for question_id={question_id}: {e}")
                out_entry = {"question_id": question_id, "prompt": prompt, "text": f"ERROR: {e}"}
                out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            resp_content = extract_response_content(resp_json)

            out_entry = {
                "question_id": question_id,
                "prompt": prompt,
                "text": resp_content,
                "flip_vartheta": _flip_file_cached_value
            }
            print(f"[INFO] line {lineno}: question_id={question_id}, response='{resp_content}'")
            out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    out_f.close()
    print(f"Done. Wrote responses to {args.output}")


if __name__ == "__main__":
    main()
