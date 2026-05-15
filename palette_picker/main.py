import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from rich.console import Console, Group
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.live import Live

# We need to import Selection from a way that works with Rich, 
# but Rich doesn't have a built-in "Select" prompt like some others.
# I will implement a simple selectable menu using Rich's layout or a table.
# Since the instructions say "interactive palette menu must use rich — no input() prompts for menu selection",
# I'll implement a custom selectable list.

PALETTES = {
    "Nord": ["#2E3440", "#3B4252", "#434C5E", "#4C566A", "#D8DEE9", "#E5E9F0", "#ECEFF4", "#8FBCBB", "#88C0D0", "#81A1C1", "#5E81AC", "#BF616A", "#D08770", "#EBCB8B", "#A3BE8C", "#B48EAD"],
    "Catppuccin Mocha": ["#1E1E2E", "#181825", "#313244", "#45475A", "#585B70", "#CDD6F4", "#BAC2DE", "#A6ADC8", "#F5E0DC", "#F2CDCD", "#F5C2E7", "#CBA6F7", "#F38BA8", "#EBA0AC", "#FAB387", "#FAE3B0", "#A6E3A1", "#94E2D5", "#89DCEB", "#89B4FA", "#74C7EC"],
    "Gruvbox": ["#282828", "#3C3836", "#504945", "#665C54", "#7C6F64", "#928374", "#A89984", "#FBF1C7", "#EBDBB2", "#D5C4A1", "#CC241D", "#FB4934", "#98971A", "#B8BB26", "#D79921", "#FABD2F", "#458588", "#83A598", "#B16286", "#D3869B", "#689D6A", "#8EC07C"],
    "Dracula": ["#282A36", "#44475A", "#F8F8F2", "#6272A4", "#8BE9FD", "#50FA7B", "#FFB86C", "#FF79C6", "#BD93F9", "#FF5555", "#F1FA8C"],
    "Solarized Dark": ["#002B36", "#073642", "#586E75", "#657B83", "#839496", "#93A1A1", "#EEE8D5", "#FDF6E3", "#B58900", "#CB4B16", "#DC322F", "#D33682", "#6C71C4", "#268BD2", "#2AA198", "#859900"],
    "Monokai": ["#272822", "#75715E", "#F8F8F2", "#F92672", "#FD971F", "#E6DB74", "#A6E22E", "#66D9EF", "#AE81FF", "#F8F8F0"],
    "One Dark": ["#282C34", "#3E4451", "#ABB2BF", "#E06C75", "#98C379", "#E5C07B", "#61AFEF", "#C678DD", "#56B6C2", "#5C6370"],
    "Tokyo Night": ["#1A1B26", "#16161E", "#1F2335", "#24283B", "#292E42", "#565F89", "#737AA2", "#A9B1D6", "#C0CAF5", "#9ECE6A", "#FF9E64", "#F7768E", "#BB9AF7", "#7DCFFF", "#2AC3DE", "#7AA2F7"],
    "Kanagawa": ["#1F1F28", "#16161D", "#2A2A37", "#223249", "#727169", "#DCD7BA", "#938AA9", "#C8C093", "#FF5D62", "#FF9E3B", "#C0A36E", "#76946A", "#7FB4CA", "#7AA89F", "#957FB8", "#D27E99"],
    "Cyberpunk": ["#000000", "#0A0A0A", "#1A1A2E", "#16213E", "#0F3460", "#533483", "#E94560", "#FF2A6D", "#05D9E8", "#D1F7FF", "#005678", "#01C5C4", "#FF4365", "#FFFFFF"],
    "GameBoy": ["#0F380F", "#306230", "#8BAC0F", "#9BBC0F"],
}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

def get_palette_slug(name):
    import re
    slug = name.lower().replace(" ", "_")
    return re.sub(r'[^a-z0-9_]', '', slug)

def process_image(image_path, palette_name, diffusion_strength):
    console = Console()
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = np.array(img, dtype=np.float32)
    
    palette_colors = np.array([hex_to_rgb(c) for c in PALETTES[palette_name]], dtype=np.float32)
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Dithering...", total=height)
        
        for y in range(height):
            for x in range(width):
                old_pixel = pixels[y, x].copy()
                
                # Vectorized distance to all palette colors
                distances = np.linalg.norm(palette_colors - old_pixel, axis=1)
                best_color = palette_colors[np.argmin(distances)]
                
                pixels[y, x] = best_color
                
                quant_error = (old_pixel - best_color) * diffusion_strength
                
                if x + 1 < width:
                    pixels[y, x + 1] += quant_error * (7 / 16)
                if y + 1 < height:
                    if x > 0:
                        pixels[y + 1, x - 1] += quant_error * (3 / 16)
                    pixels[y + 1, x] += quant_error * (5 / 16)
                    if x + 1 < width:
                        pixels[y + 1, x + 1] += quant_error * (1 / 16)
            
            progress.update(task, advance=1)

    # Clip values to [0, 255] and convert to uint8
    final_pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(final_pixels)

def select_palette_rich():
    console = Console()
    options = list(PALETTES.keys())
    
    while True:
        table = Table(title="Select a Palette", show_header=False, box=None)
        for i, name in enumerate(options, 1):
            table.add_row(f"[bold cyan]{i}[/bold cyan]", name)
        
        console.print(Panel(table, expand=False))
        choice = Prompt.ask("Enter the number of the palette")
        
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        console.print("[red]Invalid selection. Please try again.[/red]")

def main():
    parser = argparse.ArgumentParser(description="Palette Picker CLI")
    parser.add_argument("-i", "--image", help="Path to input image")
    parser.add_argument("-p", "--palette", help="Palette name")
    parser.add_argument("-d", "--diffusion", type=float, default=0.75, help="Diffusion strength (0.0-1.0)")
    
    args = parser.parse_args()
    console = Console()

    if args.image:
        if not args.palette:
            console.print("[red]Error: --palette is required in headless mode.[/red]")
            sys.exit(1)
        if args.palette not in PALETTES:
            console.print(f"[red]Error: Palette '{args.palette}' not found.[/red]")
            sys.exit(1)
        if not (0.0 <= args.diffusion <= 1.0):
            parser.error("Argument --diffusion must be between 0.0 and 1.0")
        
        image_path = args.image
        palette_name = args.palette
        diffusion = args.diffusion
    else:
        image_input = Prompt.ask("Drag and drop image path here")
        image_path = image_input.strip().strip('"').strip("'")
        
        if not os.path.exists(image_path):
            console.print("[red]Error: File not found.[/red]")
            sys.exit(1)
            
        palette_name = select_palette_rich()
        diffusion = 0.75

    try:
        result_img = process_image(image_path, palette_name, diffusion)
        
        original_path = Path(image_path)
        slug = get_palette_slug(palette_name)
        output_filename = f"{original_path.stem}_{slug}.png"
        output_path = Path.home() / "Downloads" / output_filename
        
        result_img.save(output_path)
        console.print(f"[bold green]Success![/bold green] Image saved to: [yellow]{output_path}[/yellow]")
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
