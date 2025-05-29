import huggingface_hub

model_name_simple = "multilingual-e5-base"
model_id = "intfloat/multilingual-e5-base"
local_dir = f"/mnt/d/aimodel/{model_name_simple}"

print(f"Downloading model {model_name_simple}({model_id}) to {local_dir}...")

huggingface_hub.snapshot_download(
    model_id, 
    local_dir=local_dir, 
    local_dir_use_symlinks = False
)

print(f"model {model_name_simple}({model_id}) is downloaded to {local_dir} successfully.")
