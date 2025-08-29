# Convert 4-channel CSV-Filter tensors (.pt) into grayscale-coded RGB PNGs.
# Testing and usage:
#   python gen_RGB_images.py --input_dir tensors_data --output_dir images_data
# Written for Group4 Hackathon to inspect CSV-Filter based image generation outputs visually. 

import argparse
import os
from typing import List, Tuple, Union

import torch
import numpy as np
from PIL import Image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 4-channel CSV-Filter tensors (.pt) into grayscale-coded RGB PNGs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="tensors_data",
        help="Directory containing .pt tensor files (default: tensors_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images_data",
        help="Directory to write PNG images (default: images_data)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images if present.",
    )
    return parser.parse_args()


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_tensor_payload(pt_path: str) -> Union[torch.Tensor, List, Tuple, dict]:
    payload = torch.load(pt_path, map_location="cpu")
    return payload


def extract_tensor_images(payload: Union[torch.Tensor, List, Tuple, dict]) -> List[torch.Tensor]:
    """
    Extract image tensor(s) from various payload structures found in this repo.

    Supported payloads:
    - dict with key 'image' -> tensor of shape [C,H,W] or [N,C,H,W]
    - list/tuple where first element is a tensor [C,H,W] (e.g., [image, label])
    - raw tensor of shape [C,H,W] or [N,C,H,W]
    Returns a list of tensors each shaped [C,H,W].
    """
    images: List[torch.Tensor] = []

    if isinstance(payload, dict) and "image" in payload:
        if torch.is_tensor(payload["image"]):
            tensor = payload["image"]
        elif isinstance(payload["image"], np.ndarray):
            tensor = torch.from_numpy(payload["image"])
        else:
            return images
    elif isinstance(payload, (list, tuple)) and len(payload) > 0:
        if torch.is_tensor(payload[0]):
            tensor = payload[0]
        elif isinstance(payload[0], np.ndarray):
            tensor = torch.from_numpy(payload[0])
        else:
            return images
    elif torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, np.ndarray):
        tensor = torch.from_numpy(payload)
    else:
        return images

    if tensor.dim() == 4:  # [N, C, H, W]
        for index in range(tensor.size(0)):
            images.append(tensor[index].detach().cpu())
    elif tensor.dim() == 3:  # [C, H, W]
        images.append(tensor.detach().cpu())
    elif tensor.dim() == 2:  # [H, W] (single-channel stored without C dimension)
        images.append(tensor.unsqueeze(0).detach().cpu())  # -> [1,H,W]
    else:
        # Unsupported rank
        pass

    return images


def convert_four_channel_to_gray_codes(image_4ch: torch.Tensor) -> torch.Tensor:
    """
    Convert a 4-channel [4,H,W] tensor to a single-channel grayscale-coded map per spec:
    Match=1, Deletion=2, Insertion=3, Soft-clip=4; background=0.

    Uses argmax across channels to choose the dominant operation per pixel.
    If all-channel sum at a pixel is zero, sets that pixel to 0.
    """
    assert image_4ch.dim() == 3 and image_4ch.size(0) == 4, "Expected [4,H,W] tensor"

    image_float = image_4ch.float()
    per_pixel_sum = image_float.sum(dim=0)  # [H,W]
    label_map = image_float.argmax(dim=0) + 1  # 1..4
    label_map[per_pixel_sum == 0] = 0  # background where no signal
    return label_map.to(torch.uint8)


def save_gray_codes_as_rgb_png(gray_codes: torch.Tensor, output_png_path: str) -> None:
    """
    Replicate grayscale-coded map into 3-channel RGB and save as PNG.

    To make codes visually discernible, map labels {0,1,2,3,4}
    to brightness levels {0,64,128,192,255} before saving.
    """
    if gray_codes.dtype != torch.uint8:
        gray_codes = gray_codes.to(torch.uint8)

    # Map codes to visible brightness levels
    levels = torch.tensor([0, 64, 128, 192, 255], dtype=torch.uint8)
    gray_index = gray_codes.clamp(min=0, max=4).to(torch.long)
    gray_vis = levels[gray_index]

    # Convert to numpy for PIL
    gray_vis_np = gray_vis.numpy()
    image_l = Image.fromarray(gray_vis_np, mode="L")
    image_rgb = image_l.convert("RGB")
    image_rgb.save(output_png_path)


def process_file(pt_path: str, output_dir: str, overwrite: bool) -> List[str]:
    payload = load_tensor_payload(pt_path)
    tensors = extract_tensor_images(payload)
    written: List[str] = []

    if not tensors:
        return written

    base_name = os.path.splitext(os.path.basename(pt_path))[0]

    for index, image_tensor in enumerate(tensors):
        if image_tensor.size(0) == 4:
            gray_codes = convert_four_channel_to_gray_codes(image_tensor)
        elif image_tensor.size(0) == 1:
            # Already single-channel; keep values as-is but ensure uint8
            gray_codes = image_tensor[0].clamp(min=0, max=255).to(torch.uint8)
        else:
            # Unsupported channel count; skip
            continue

        out_name = f"{base_name}.png" if len(tensors) == 1 else f"{base_name}_{index}.png"
        out_path = os.path.join(output_dir, out_name)
        if not overwrite and os.path.exists(out_path):
            written.append(out_path)
            continue

        save_gray_codes_as_rgb_png(gray_codes, out_path)
        written.append(out_path)

    return written


def main() -> None:
    args = parse_arguments()
    ensure_directory(args.output_dir)

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    pt_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".pt")
    ]

    if not pt_files:
        print(f"No .pt files found in {args.input_dir}")
        return

    total_written = 0
    for pt_path in sorted(pt_files):
        try:
            outputs = process_file(pt_path, args.output_dir, args.overwrite)
            total_written += len(outputs)
            for out_path in outputs:
                print(f"Wrote {out_path}")
        except Exception as exc:
            print(f"Failed {pt_path}: {exc}")

    print(f"Done. Total images written: {total_written}")


if __name__ == "__main__":
    main()
