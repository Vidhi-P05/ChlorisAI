from huggingface_hub import upload_file

# Path to your local model
local_model_path = "checkpoints/best_model.pth"

# The filename in the Hugging Face repo
remote_model_name = "best_model.pth"

# Your Hugging Face repo
repo_id = "Vidhi-Pateliya-01/ChlorisAI-model"  

# Upload the file
upload_file(
    path_or_fileobj=local_model_path,
    path_in_repo=remote_model_name,
    repo_id=repo_id,
    repo_type="model",
)
