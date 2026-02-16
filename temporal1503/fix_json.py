import json
import os

def repair_dataset_json():
    # --- 1. CONFIGURATION (Update these to match your current folders) ---
    
    # Folder containing the JSON and Images (Created by generate_dataset_final.py)
    # Check your folder structure: is it 'final_dataset', 'output_dataset', etc?
    data_dir = "final_dataset" 
    json_filename = "dataset.json" # Or "dataset_labels.json"
    
    # Folder containing the Physics/Mesh data (Created by deformation_sequential_final.py)
    mesh_dir = "output_neuro_final"
    
    # -------------------------------------------------------------------
    
    # Path construction
    json_path = os.path.join(data_dir, json_filename)
    
    if not os.path.exists(json_path):
        # Fallback check
        alt_path = os.path.join(data_dir, "dataset_labels.json")
        if os.path.exists(alt_path):
            json_path = alt_path
        else:
            print(f"CRITICAL: Could not find JSON at {json_path}")
            print(f"  -> Please check 'data_dir' variable in the script.")
            return

    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Found {len(dataset)} entries. Linking physics files...")
    
    updated_count = 0
    missing_count = 0
    
    for image_key, entry in dataset.items():
        # Source name usually looks like: "branch_00_pass_00_frame_000_mesh"
        mesh_source_name = entry.get('mesh_source')
        
        if mesh_source_name:
            # 1. Strip "_mesh" to get the base ID
            # "branch_00..._mesh" -> "branch_00..._frame_000"
            base_name = mesh_source_name.replace("_mesh", "") 
            
            # 2. Construct paths for ALL 3 required files
            # Centerline (Target Shape)
            cl_filename = f"{base_name}_centerline.npy"
            cl_path = os.path.join(mesh_dir, cl_filename)
            
            # Sparse Displacements (Physics Parameters) - NEW REQUIREMENT
            sd_filename = f"{base_name}_sparse_disp.npy"
            sd_path = os.path.join(mesh_dir, sd_filename)
            
            # Control Point Indices (Physics Locations) - NEW REQUIREMENT
            cp_filename = f"{base_name}_cp_indices.npy"
            cp_path = os.path.join(mesh_dir, cp_filename)
            
            # 3. Verify and Link
            if os.path.exists(cl_path):
                entry['centerline_file'] = cl_path
                
                # Link sparse data if it exists (Critical for RBF-PINN)
                if os.path.exists(sd_path):
                    entry['sparse_disp_file'] = sd_path
                if os.path.exists(cp_path):
                    entry['cp_indices_file'] = cp_path
                    
                updated_count += 1
            else:
                # Debug info if missing
                print(f"  [MISSING] {cl_filename} not found in {mesh_dir}")
                missing_count += 1
        else:
            print(f"  [SKIP] Entry {image_key} has no 'mesh_source'")

    # Save the Fixed JSON
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print("-" * 30)
    print(f"REPAIR COMPLETE")
    print(f"Updated: {updated_count} entries")
    print(f"Missing: {missing_count} entries")
    print("-" * 30)
    print("Next: Run 'train.py'. It will now find 'sparse_disp_file' correctly.")

if __name__ == "__main__":
    repair_dataset_json()