import os
import sys
import argparse
import numpy as np
from PIL import Image
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import track

console = Console()

PALETTES = {
    "Dracula": [
        [40, 42, 54], [68, 71, 90], [248, 248, 242], [98, 114, 164],
        [139, 233, 253], [80, 250, 123], [255, 184, 108], [255, 121, 198],
        [189, 147, 249], [255, 85, 85], [241, 250, 140]
    ],
    "Gray Scale 1 bit (Black & White)": [
        [0, 0, 0], [255, 255, 255]
    ],
    "Gray Scale 2 bits": [
        [0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]
    ],
    "Gray, 4 shades, gamma-corrected": [
        [0, 0, 0], [122, 122, 122], [186, 186, 186], [255, 255, 255]
    ],
    "Gruvbox": [
        [40, 40, 40], [60, 56, 54], [80, 73, 69], [102, 92, 84],
        [124, 111, 100], [235, 219, 178], [251, 241, 199], [213, 196, 161],
        [189, 174, 147], [168, 153, 132], [204, 36, 29], [254, 128, 25],
        [215, 153, 33], [152, 151, 26], [104, 157, 106], [211, 134, 155],
        [131, 165, 152], [142, 192, 124]
    ],
    "Ice Cream GB": [
        [224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]
    ],
    "Kanagawa": [
        [31, 31, 40], [42, 42, 55], [54, 54, 70], [84, 84, 109],
        [220, 215, 186], [193, 192, 176], [114, 113, 105], [195, 64, 67],
        [232, 36, 36], [118, 148, 106], [152, 187, 108], [192, 163, 110],
        [230, 195, 132], [126, 156, 216], [127, 180, 202], [149, 127, 184],
        [147, 140, 170], [106, 149, 137]
    ],
    "Kirokaze GameBoy": [
        [51, 44, 80], [70, 135, 143], [148, 227, 68], [226, 243, 228]
    ],
    "Monokai": [
        [39, 40, 34], [56, 56, 48], [73, 72, 62], [117, 113, 94],
        [165, 159, 133], [248, 248, 242], [245, 244, 241], [249, 38, 114],
        [253, 151, 31], [230, 219, 116], [166, 226, 46], [102, 217, 239],
        [174, 129, 255]
    ],
    "Nord": [
        [46, 52, 64], [59, 66, 82], [67, 76, 94], [76, 86, 106],
        [216, 222, 233], [229, 233, 240], [236, 239, 244], [143, 188, 187],
        [136, 192, 208], [129, 161, 193], [94, 129, 172], [191, 97, 106],
        [208, 135, 112], [235, 203, 139], [163, 190, 140], [180, 142, 173]
    ],
    "Catppuccin Mocha": [
        [30, 30, 46], [24, 24, 37], [17, 17, 27], [205, 214, 244],
        [245, 224, 220], [242, 205, 205], [245, 194, 231], [203, 166, 247],
        [243, 139, 168], [235, 160, 172], [250, 179, 135], [249, 226, 175],
        [166, 227, 161], [148, 226, 213], [137, 220, 235], [116, 199, 236],
        [137, 180, 250], [180, 190, 254]
    ]
}

def get_save_path(original_filename, palette_name):
    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    if not os.path.exists(downloads_path):
        downloads_path = os.getcwd()
    
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    safe_palette_name = palette_name.replace(" ", "_").replace("&", "and").replace(",", "").lower()
    new_filename = f"{base_name}_{safe_palette_name}.png"
    
    return os.path.join(downloads_path, new_filename)

def apply_floyd_steinberg(image_path, palette_name, diffusion_strength=0.75):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        console.print(f"[bold red]Error opening image:[/bold red] {e}")
        sys.exit(1)

    img_array = np.array(img, dtype=float)
    height, width, _ = img_array.shape
    palette_array = np.array(PALETTES[palette_name], dtype=float)

    # Catppuccin themed progress bar
    for y in track(range(height), description=f"[cyan]Applying {palette_name} palette (Smoothness: {1.0-diffusion_strength:.2f})..."):
        for x in range(width):
            # Clamp before processing to keep colors grounded
            old_pixel = np.clip(img_array[y, x], 0, 255)
            
            # Find closest palette color
            distances = np.sum((palette_array - old_pixel) ** 2, axis=1)
            new_pixel = palette_array[np.argmin(distances)]
            
            # Apply color and calculate quantized error
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Apply diffusion strength multiplier to reduce grain
            spreadable_error = quant_error * diffusion_strength

            # Diffuse the error to neighbors
            if x + 1 < width:
                img_array[y, x + 1] += spreadable_error * (7 / 16)
            if x - 1 >= 0 and y + 1 < height:
                img_array[y + 1, x - 1] += spreadable_error * (3 / 16)
            if y + 1 < height:
                img_array[y + 1, x] += spreadable_error * (5 / 16)
            if x + 1 < width and y + 1 < height:
                img_array[y + 1, x + 1] += spreadable_error * (1 / 16)

    # Clamp final output and convert back to image
    final_image_array = np.clip(img_array, 0, 255).astype(np.uint8)
    final_image = Image.fromarray(final_image_array, 'RGB')
    
    save_path = get_save_path(image_path, palette_name)
    final_image.save(save_path)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Apply retro color palettes using Floyd-Steinberg dithering.")
    parser.add_argument("-i", "--image", type=str, help="Path to the input image")
    parser.add_argument("-p", "--palette", type=str, choices=list(PALETTES.keys()), help="Name of the palette to apply")
    # Added optional diffusion flag (default 0.75 for smoothness)
    parser.add_argument("-d", "--diffusion", type=float, default=0.75, help="Strength of error diffusion (0.0 to 1.0). Lower is smoother. Default 0.75.")
    
    args = parser.parse_args()

    # Vibe Check / Aesthetics
    console.print(Panel.fit("[bold magenta]Palette Picker CLI[/bold magenta]", border_style="cyan"))

    # Argument validation
    diffusion = np.clip(args.diffusion, 0.0, 1.0)

    if args.image and args.palette:
        image_path = args.image
        selected_palette = args.palette
        if not os.path.exists(image_path):
            console.print("[bold red]File not found. Please check the path.[/bold red]")
            sys.exit(1)
    else:
        # Interactive Mode
        image_path = Prompt.ask("[bold green]Drag and drop your image here (or paste path)[/bold green]").strip("'\" ")
        
        if not os.path.exists(image_path):
            console.print("[bold red]File not found. Please check the path.[/bold red]")
            sys.exit(1)

        console.print("\n[bold cyan]Available Palettes:[/bold cyan]")
        palette_names = list(PALETTES.keys())
        for i, name in enumerate(palette_names, 1):
            console.print(f"[{i}] {name}")
        
        while True:
            try:
                choice = int(Prompt.ask("\n[bold green]Select a palette number[/bold green]"))
                if 1 <= choice <= len(palette_names):
                    selected_palette = palette_names[choice - 1]
                    break
                else:
                    console.print("[red]Invalid choice. Try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")

    console.print()
    # Apply processing with new diffusion strength
    output_path = apply_floyd_steinberg(image_path, selected_palette, diffusion)
    console.print(f"\n[bold green]Done![/bold green] Image saved to:\n[blue]{output_path}[/blue]")

if __name__ == "__main__":
    main()