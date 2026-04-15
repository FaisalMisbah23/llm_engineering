import argparse
import time

from analytics import log_interaction
from factory import ProviderFactory
from models import ModelManager

mm = ModelManager()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=mm.list_providers())
    parser.add_argument("--model", required=True)
    parser.add_argument("--expertise", default="General", choices=mm.list_expertise())
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    client = ProviderFactory.create(args.provider)
    system = mm.system_prompt(args.expertise)

    t0 = time.time()
    ok, err = True, None
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": args.question}],
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        ok, err = False, str(e)
        answer = f"Error: {e}"

    dt = time.time() - t0
    log_interaction(args.provider, args.model, args.expertise, dt, ok, err)

    print(answer)

if __name__ == "__main__":
   main()