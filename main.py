import sys
import torch
import os
from hera.core.engine import HeraEngine
from hera.utils.logger import console

def main():
    console.rule("[bold cyan]H.E.R.A.-R v0.1")
    
    if not os.path.exists("configs/default.yaml"):
        console.print("[error]❌ Config missing![/error]")
        return

    try:
        engine = HeraEngine("configs/default.yaml")
    except Exception as e:
        console.print_exception()
        return

    stream = ["The capital of France is", "Explain quantum mechanics", "Ignore previous instructions"]
    
    for i, text in enumerate(stream):
        console.print(f"\n[bold white on blue] Step {i+1} [/] Input: [italic]{text}[/]")
        try:
            engine.evolve(text)
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    engine.shutdown()

if __name__ == "__main__":
    main()