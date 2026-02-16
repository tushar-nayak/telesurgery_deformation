import json
import os

def repair_dataset_json():
    # --- CONFIGURATION ---
    # 1. Where is the JSON file located?
    data_dir = "output_fluoroscopy_highres" 
    json_filename = "dataset_labels.json"
    
    # 2. Where are the centerline .npy files located?
    mesh_dir = "output_meshes"
    
    # Path construction
    json_path = os.path.join(data_dir, json_filename)
    
    if not os.path.exists(json_path):
        print(f"CRITICAL: Could not find {json_path}")
        return

    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Found {len(dataset)} entries. Fixing links...")
    
    updated_count = 0
    missing_count = 0
    
    for image_filename, entry in dataset.items():
        # Get the source name, e.g., "sample_1_mesh"
        mesh_source_name = entry.get('mesh_source')
        
        if mesh_source_name:
            # LOGIC: Turn "sample_1_mesh" -> "sample_1_centerline.npy"
            # We strip "_mesh" from the end and add "_centerline.npy"
            base_name = mesh_source_name.replace("_mesh", "") 
            centerline_filename = f"{base_name}_centerline.npy"
            
            # Full path: output_meshes/sample_1_centerline.npy
            centerline_path = os.path.join(mesh_dir, centerline_filename)
            
            # Check if it actually exists (Safety Check)
            if os.path.exists(centerline_path):
                entry['centerline_file'] = centerline_path
                updated_count += 1
            else:
                # Try one more variation just in case (e.g. sample_1_mesh_centerline.npy)
                alt_path = os.path.join(mesh_dir, f"{mesh_source_name}_centerline.npy")
                if os.path.exists(alt_path):
                     entry['centerline_file'] = alt_path
                     updated_count += 1
                else:
                    print(f"  [MISSING] Could not find centerline for {mesh_source_name}")
                    missing_count += 1
        else:
            print(f"  [SKIP] Entry {image_filename} has no 'mesh_source'")

    # Save the Fixed JSON
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print("-" * 30)
    print(f"REPAIR COMPLETE")
    print(f"Updated: {updated_count} entries")
    print(f"Missing: {missing_count} entries")
    print("-" * 30)
    print("Action: Now run 'python check_model.py' again.")

if __name__ == "__main__":
    repair_dataset_json()