"""Pack the MCP server as a Claude Desktop extension: dist/jev-browser.mcpb. No credentials are included."""

import json
import tomllib
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "packaging" / "mcpb"
FILES = ["pyproject.toml", "uv.lock", "README.md", "LICENSE"]


def icon():
    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 16, 496, 496), radius=112, fill=(24, 32, 28, 255))
    draw.polygon([(292, 72), (140, 292), (244, 292), (212, 440), (372, 212), (268, 212)], fill=(250, 204, 21, 255))
    path = ROOT / "dist" / "icon.png"
    image.save(path)
    return path


def main():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if manifest["version"] != version:
        raise SystemExit(f"manifest version {manifest['version']} != pyproject version {version}")
    (ROOT / "dist").mkdir(exist_ok=True)
    out = ROOT / "dist" / "jev-browser.mcpb"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(BUNDLE / "manifest.json", "manifest.json")
        bundle.write(BUNDLE / "server.py", "server.py")
        bundle.write(icon(), "icon.png")
        for name in FILES:
            bundle.write(ROOT / name, name)
        for path in sorted((ROOT / "jev_ultrafast").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                bundle.write(path, path.relative_to(ROOT).as_posix())
    print(out)


if __name__ == "__main__":
    main()
