# tests/test_parser.py
from app.services.genAI.rag.parser import parse_repo
import os

def test_parser_chunks_single_file():
    repo_path = os.path.join("D:/CodePilot/repos/react-practices/src/api")
    chunks = parse_repo(repo_path, extensions=[".ts", ".tsx"])
    print([chunk['file'] for chunk in chunks])    
    assert len(chunks) > 0
    for c in chunks:
        assert "chunk" in c
        assert "file" in c
