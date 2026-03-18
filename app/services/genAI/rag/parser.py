import os

skip_list = ['node_modules', '.github']
CHUNK_SIZE = 50  # liczba linii w jednym chunku

def parse_repo(repo_path: str, extensions=None):
    if extensions is None:
        extensions = ['.ts', '.tsx', '.js', '.jsx', '.json', '.html', '.css', '.scss']
       
    chunks = []   

    for root, _, files in os.walk(repo_path):
        if any(skip in root for skip in skip_list):
            continue

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")
                    for i in range(0, len(lines), CHUNK_SIZE):
                        chunk = "\n".join(lines[i:i+CHUNK_SIZE])
                        chunks.append({
                        "metadata": {
                            "file_path": os.path.relpath(os.path.join(root, file), repo_path),
                            "file_name": file,
                            "folder": os.path.relpath(root, repo_path)
                        }, 
                        "chunk": chunk
                    })
    return chunks
            