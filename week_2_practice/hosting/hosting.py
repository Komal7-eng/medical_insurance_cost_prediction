from huggingface_hub import HfApi
import os
# 🔥 LOAD TOKEN
load_dotenv()


api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="week_2_practice/deployment",     # the local folder containing your files
    repo_id="kaushalya7/medical-insurance-cost-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
