import os
import shutil
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
import torch.nn as nn


def copy_code_folder(output_dir="code_clean", skip_dirs=["data", "workspace", "sapiens", ".git"]):
    """Copy current code folder to a new directory following `.gitignore` rules.
    Large dataset folders listed in `skip_dirs` will be skipped.
    """
    root_dir = os.getcwd()
    gitignore_path = os.path.join(root_dir, ".gitignore")

    # Read .gitignore rules
    spec = None
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            spec = PathSpec.from_lines(GitWildMatchPattern, f)

    # Normalize skip directory paths
    skip_dirs_abs = set()
    if skip_dirs:
        for d in skip_dirs:
            skip_dirs_abs.add(os.path.abspath(os.path.normpath(d)))

    # Create output directory
    output_dir_abs = os.path.abspath(os.path.normpath(output_dir))
    os.makedirs(output_dir_abs, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_dir_abs = os.path.abspath(os.path.normpath(dirpath))

        # Skip output directory
        if current_dir_abs == output_dir_abs:
            print(f"Skipping output directory: {dirpath}")
            dirnames[:] = []
            continue

        # Skip dataset directories
        if current_dir_abs in skip_dirs_abs:
            print(f"Skipping dataset directory: {dirpath}")
            dirnames[:] = []
            continue

        # Iterate files
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(file_path, root_dir)

            # Skip files matched by .gitignore
            if spec and spec.match_file(rel_path):
                print(f"Ignored (gitignore): {rel_path}")
                continue

            # Destination path
            dest_path = os.path.join(output_dir_abs, rel_path)
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)

            # Copy file
            shutil.copy2(file_path, dest_path)
            print(f"Copied: {rel_path}")

    print(f"Copy complete, files saved to {output_dir_abs}")


def count_parameters(model: nn.Module, detailed: bool = True) -> int:
    """Count parameters in a PyTorch model.

    Args:
        model: PyTorch model to count parameters for
        detailed: whether to print per-layer details

    Returns:
        Total number of parameters in the model
    """
    from prettytable import PrettyTable
    total_params = 0
    
    if detailed:
        table = PrettyTable(["Layer Name", "Param Count", "Trainable"])
    
    for name, parameter in model.named_parameters():
        if "img_encoder" in name:
            continue
        if "vol_decoder" in name:
            continue
        param_count = parameter.numel()
        total_params += param_count
        
        if detailed:
            table.add_row([name, param_count, parameter.requires_grad])
    
    if detailed:
        print(table)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return total_params


if __name__ == "__main__":
    copy_code_folder(output_dir="code_clean")
