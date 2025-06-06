import os
import numpy as np
from glob import glob

def unpack_npz_batch(input_dir, output_root_dir):
    os.makedirs(output_root_dir, exist_ok=True)

    # Find all env_* directories
    env_dirs = sorted(glob(os.path.join(input_dir, "env_*")))

    for env_dir in env_dirs:
        env_name = os.path.basename(env_dir)
        output_env_dir = os.path.join(output_root_dir, env_name)
        os.makedirs(output_env_dir, exist_ok=True)

        # Find all episode_*.npz files in the current env directory
        npz_files = sorted(glob(os.path.join(env_dir, "episode_*.npz")))

        for npz_file in npz_files:
            episode_name = os.path.splitext(os.path.basename(npz_file))[0]
            episode_dir = os.path.join(output_env_dir, episode_name)
            os.makedirs(episode_dir, exist_ok=True)

            # Load .npz
            data = np.load(npz_file)
            for key in data.files:
                out_path = os.path.join(episode_dir, f"{key}.npy")
                np.save(out_path, data[key])
            
            print(f"Unpacked {episode_name} to {episode_dir}")

if __name__ == "__main__":
    # Example usage
    input_root = "./inference_logs_2_1/inference_logs/20250523_012807"     # Root directory containing env_* directories
    output_root = "./organized_data"   # Target root directory
    unpack_npz_batch(input_root, output_root)
