import ollama
import argparse
import sys
try:
  import readline
except:
  pass

parser = argparse.ArgumentParser()
parser.add_argument("--system", help="Set system message", default=None)
parser.add_argument("--num_ctx", help="Set context size", default=None)
parser.add_argument("--temperature", help="Set tempaeratur", default=None)
parser.add_argument("--nostream", help="Disable streaming", default=False, action="store_true")
parser.add_argument("model")
parser.add_argument("prompts", nargs='*')
args = parser.parse_args()

client = ollama.Client()
userprompt = ">>> " if sys.stdin.isatty() else ""

options = {}
if args.temperature:
  options["temperature"] = args.temperature
if args.num_ctx:
  options["num_ctx"] = int(args.num_ctx)

def chat(messages, prompt):
  messages.append({"role":"user", "content": prompt})
  response = client.chat(model=args.model, messages=messages, options=options, stream=not args.nostream)
  m = ''
  for r in response if not args.nostream else [response]:
    c = r['message']['content']
    print(c, end='', flush=True)
    m = m + c
  print()
  messages.append({"role": "assistant", "content": m})
  return messages

messages = []
if args.system:
  messages.append({"role":"system","content":args.system})
for prompt in args.prompts:
  messages = chat(messages, prompt)
while True:
  try:
    prompt = input(userprompt)
  except:
    print()
    break
  if prompt == "/bye":
    break
  messages = chat(messages, prompt)