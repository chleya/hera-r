from rich.console import Console
from rich.theme import Theme

# 自定义配色主题
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

    def info(self, msg):
        if self.verbose:
            console.print(f"[info]ℹ️  {msg}[/info]")

    def step(self, msg):
        console.print(f"[step]🔄 {msg}[/step]")

    def success(self, msg):
        console.print(f"[success]✅ {msg}[/success]")

    def warning(self, msg):
        console.print(f"[warning]⚠️  {msg}[/warning]")

    def error(self, msg):
        console.print(f"[error]❌ {msg}[/error]")
        
    def exception(self, msg):
        console.print_exception(show_locals=True) # 显示详细报错现场