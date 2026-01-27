from rich.console import Console
from rich.theme import Theme
from datetime import datetime

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "step": "magenta"
})

console = Console(theme=custom_theme)

class HeraLogger:
    def __init__(self, cfg):
        self.verbose = cfg.get("logging", {}).get("verbose", True)

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S")

    def info(self, msg):
        if self.verbose:
            console.print(f"[dim]{self._ts()}[/dim] [info]ℹ️  {msg}[/info]")

    def step(self, msg):
        console.print(f"[dim]{self._ts()}[/dim] [step]🔄 {msg}[/step]")

    def success(self, msg):
        console.print(f"[dim]{self._ts()}[/dim] [success]✅ {msg}[/success]")

    def warning(self, msg):
        console.print(f"[dim]{self._ts()}[/dim] [warning]⚠️  {msg}[/warning]")

    def error(self, msg):
        console.print(f"[dim]{self._ts()}[/dim] [error]❌ {msg}[/error]")