# 🎨 Palette Picker CLI

Turn your photos into retro, dithered art with a single command. 

Palette Picker CLI is a lightweight tool that uses Floyd-Steinberg dithering to snap your images to a curated set of aesthetic color palettes. Whether you want the clean look of **Nord**, the cozy vibes of **Gruvbox**, or the classic **GameBoy** 4-color constraint, this tool has you covered.

## ✨ Features

- **Interactive Mode**: Just run the tool and follow the prompts. Drag and drop your image, pick a palette from a stylized menu, and you're done.
- **Headless Mode**: Perfect for scripting. Pass arguments for the image, palette, and diffusion strength.
- **Aesthetic Palettes**: Includes 11 predefined palettes (Nord, Catppuccin, Dracula, Tokyo Night, etc.).
- **Customizable Dithering**: Adjust the `diffusion_strength` to control how "noisy" or "smooth" the dithering looks.
- **Fast Processing**: Powered by `numpy` for vectorized color distance calculations.

## 🚀 Installation

To install the tool globally and run it from anywhere:

### Using pipx (Recommended)
```bash
# Install pipx if you haven't already
sudo pacman -S python-pipx
pipx ensurepath

# Install the tool
pipx install .
```

### Using pip (Local/Venv)
```bash
pip install .
```

## 🛠 Usage

### Interactive Mode
Simply run:
```bash
palette-picker
```
Then follow the on-screen prompts to select your image and palette.

### Headless Mode
```bash
palette-picker -i path/to/image.jpg -p "Nord" -d 0.75
```

**Arguments:**
- `-i, --image`: Path to the input image (required in headless).
- `-p, --palette`: Name of the palette to use (e.g., `"Nord"`, `"GameBoy"`) (required in headless).
- `-d, --diffusion`: Diffusion strength from `0.0` to `1.0` (default: `0.75`).

### Output
Processed images are automatically saved to your `~/Downloads` folder as `{original_name}_{palette}.png`.

## 🎨 Supported Palettes
- Nord
- Catppuccin Mocha
- Gruvbox
- Dracula
- Solarized Dark
- Monokai
- One Dark
- Tokyo Night
- Kanagawa
- Cyberpunk
- GameBoy
