import subprocess
import sys
import torch

def check_environment():
    print("=== Hardware & Environment Check ===")
    if not torch.cuda.is_available():
        print("[!] Warning: CUDA is not available. Training will run on CPU (very slow).")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[*] Detected GPU: {gpu_name} ({gpu_vram:.2f} GB VRAM)")
    
    if "3090" in gpu_name or gpu_vram > 20:
        print("[+] High-VRAM GPU detected. Optimizing settings for RTX 3090...")
        return True
    return False

def main():
    is_rtx_3090 = check_environment()
    batch_size = 4 if is_rtx_3090 else 2
    
    cmd = [
        sys.executable, 
        "exps/det/CRN_r18_256x704_128x128_4key_front_only.py",
        "--batch_size_per_device", str(batch_size),
        "--max_epochs", "24",
        "--precision", "16",
        "--accelerator", "gpu",
        "--devices", "1"
    ]
    
    print(f"\n[*] Launching training command:\n{' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Training exited with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
        sys.exit(0)

if __name__ == '__main__':
    main()
