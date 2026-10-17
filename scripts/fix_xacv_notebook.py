import json
import os
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "paper_figures.ipynb"
    print(f"Loading notebook from: {notebook_path}")
    
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
        
    modified_cell_4 = False
    modified_cell_10 = False
    
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
            
        source = cell.get("source", [])
        source_str = "".join(source)
        
        # Identify Cell 4 (first XACV visualization cell)
        if "dataset = 'xacv'" in source_str and "('u-net', 'U-Net')" in source_str:
            print(f"Found Cell 4 at index {idx}. Modifying panel_specs...")
            
            # Find the panel_specs definition and replace it
            target_specs = [
                "    ('image', 'Image'),\n",
                "    ('gt', 'GT'),\n",
                "    ('proposed', 'Proposed'),\n",
                "    ('dconnnet', 'DconnNet'),\n",
                "    ('u-net', 'U-Net'),\n",
                "    ('attention_u-net', 'Attention UNet'),\n",
                "    ('fru-net', 'FR-UNet'),\n",
                "    ('csnet', 'CS-Net'),\n",
                "    ('v-gan', 'V-GAN'),\n",
                "    ('segdiff', 'SegDiff'),\n",
                "    ('medsegdiff', 'MedSegDiff'),\n",
                "    ('cldice', 'clDice'),\n",
                "    ('cape', 'CAPE'),\n",
                "]\n"
            ]
            
            # Locate target_specs sequence in source
            start_idx = -1
            for i in range(len(source)):
                if source[i:i+len(target_specs)] == target_specs:
                    start_idx = i
                    break
            
            if start_idx != -1:
                new_specs = [
                    "    ('image', 'Image'),\n",
                    "    ('gt', 'GT'),\n",
                    "    ('proposed', 'Proposed'),\n",
                    "    ('dconnnet', 'DconnNet'),\n",
                    "]\n"
                ]
                source[start_idx:start_idx+len(target_specs)] = new_specs
                cell["source"] = source
                modified_cell_4 = True
                print("Cell 4 modified successfully.")
            else:
                print("Warning: Could not locate exact panel_specs array in Cell 4 source.")
                
        # Identify Cell 10 (second XACV overlay cell)
        elif "dataset = 'xacv'" in source_str and "target_size = (512, 512)" in source_str and "overlay = np.zeros" in source_str:
            print(f"Found Cell 10 at index {idx}. Modifying gt_bin definition...")
            
            target_segment = [
                "    image_panels = [(img, title, token) for token, title in panel_specs if (img := _pick_image(token, sample_id)) is not None]\n",
                "\n",
                "    num_images = len(image_panels)\n"
            ]
            
            start_idx = -1
            for i in range(len(source)):
                if source[i:i+len(target_segment)] == target_segment:
                    start_idx = i
                    break
                    
            if start_idx != -1:
                new_segment = [
                    "    image_panels = [(img, title, token) for token, title in panel_specs if (img := _pick_image(token, sample_id)) is not None]\n",
                    "\n",
                    "    gt_name = next((img for img, _, tok in image_panels if tok == 'gt'), None)\n",
                    "    if gt_name is not None:\n",
                    "        gt_arr = plt.imread(os.path.join(image_dir, gt_name))\n",
                    "        gt_arr = _resize_np_image(gt_arr, size=target_size, is_mask=True)\n",
                    "        gt_bin = _to_binary_2d(gt_arr)\n",
                    "    else:\n",
                    "        gt_bin = np.zeros(target_size, dtype=bool)\n",
                    "\n",
                    "    num_images = len(image_panels)\n"
                ]
                source[start_idx:start_idx+len(target_segment)] = new_segment
                cell["source"] = source
                modified_cell_10 = True
                print("Cell 10 modified successfully.")
            else:
                print("Warning: Could not locate target segment for gt_bin injection in Cell 10.")
                
    if modified_cell_4 or modified_cell_10:
        print("Saving modified notebook...")
        with notebook_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook saved successfully!")
    else:
        print("No cells were modified. Please verify match criteria.")

if __name__ == "__main__":
    main()
