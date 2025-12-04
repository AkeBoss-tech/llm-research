import os
import zipfile
import requests
from pathlib import Path
from datasets import load_dataset
from tasks.common import Task

class LILAMath(Task):
    def __init__(self, split, subset="iid", **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        lila_split = split
        
        # Data handling
        cache_dir = Path(os.path.expanduser("~/.cache/nanochat/lila"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = cache_dir / "lila.zip"
        # Check if we need to download
        if not zip_path.exists():
            print(f"Downloading LILA dataset to {zip_path}...")
            url = "https://github.com/allenai/Lila/raw/b81117ac7e56cc1dfb0fcabf0005d1755177252b/lila.zip"
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download complete.")
            except Exception as e:
                if zip_path.exists():
                    zip_path.unlink()
                raise RuntimeError(f"Failed to download LILA dataset: {e}")

        # Target file in the zip
        target_member = f"lila/multi/{subset}/{lila_split}.json"
        json_path = cache_dir / target_member
        
        if not json_path.exists():
            print(f"Extracting {target_member}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extract(target_member, path=cache_dir)
            except KeyError:
                raise ValueError(f"Could not find {target_member} in lila.zip")
            except zipfile.BadZipFile:
                # Corrupt zip, remove it
                zip_path.unlink()
                raise RuntimeError("lila.zip is corrupt, please try running again to re-download.")

        # Load dataset using the generic json loader
        # We specify split="train" because load_dataset("json") puts everything in "train" split by default
        self.ds = load_dataset("json", data_files=str(json_path), split="train")
        # Shuffle
        self.ds = self.ds.shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row['Input']
        code = row['Output Program']
        answer_list = row['Output Answer']
        
        # We use the first answer for the "assistant" message in the conversation history
        # (what we would expect the model to say ideally)
        canonical_answer = answer_list[0] if isinstance(answer_list, list) and len(answer_list) > 0 else str(answer_list)

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": canonical_answer}
        ]

        conversation = {
            "messages": messages,
            "valid_answers": answer_list if isinstance(answer_list, list) else [str(answer_list)]
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        valid_answers = conversation.get("valid_answers", [])
        if not valid_answers:
            # Fallback to checking the message content
            valid_answers = [conversation['messages'][-1]['content']]

        def normalize(text):
            return text.strip()

        response = normalize(assistant_response)
        
        for ans in valid_answers:
            ans = normalize(ans)
            if response == ans:
                return 1
            # Try numeric comparison
            try:
                if abs(float(response) - float(ans)) < 1e-7:
                    return 1
            except (ValueError, TypeError):
                pass
                
        return 0
