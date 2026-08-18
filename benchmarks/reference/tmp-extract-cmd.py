"""Extrae los comandos completos de llama-eval-callback del transcript (temporal)."""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
path = r"C:\Users\gpsan\.claude\projects\D--AI-projects-unltd-inference\7728faf4-d66f-4c33-bc0b-92560e694ca0.jsonl"
text = open(path, encoding="utf-8", errors="replace").read()
seen = set()
for m in re.finditer(r"cd /d/AI/runtimes/llama\.cpp/build && MSBuild\.exe(?:\\.|[^\\])*", text):
    chunk = text[m.start() - 20 : m.start() + 1600]
    chunk = chunk.replace('\\"', '"')
    chunk = chunk.replace("\\n", "\n")
    chunk = chunk.replace("\\\\", "\\")
    if "llama-eval-callback.exe -m" in chunk and chunk not in seen:
        seen.add(chunk)
        print(chunk[:1450])
        print("======")
