import os
import tempfile
import shutil
from typing import List
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from urllib.parse import urlparse

def parse_github_url(url: str) -> tuple:
    """
    Parse GitHub URL to extract owner and repo name.
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Tuple of (owner, repo_name)
    """
    # Remove .git suffix if present
    url = url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]
    
    # Parse the URL
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    else:
        raise ValueError("Invalid GitHub URL format")

def load_github_repo(repo_url: str, branch: str = "main") -> List[Document]:
    """
    Clone a GitHub repository and load its contents as documents.
    
    Args:
        repo_url: GitHub repository URL
        branch: Branch name to clone (default: main)
        
    Returns:
        List of Document objects containing the repository content
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Clone the repository
        print(f"Cloning repository: {repo_url}")
        try:
            repo = Repo.clone_from(repo_url, temp_dir, branch=branch, depth=1)
        except Exception as e:
            # Try with 'master' branch if 'main' fails
            if branch == "main":
                print("'main' branch not found, trying 'master'...")
                repo = Repo.clone_from(repo_url, temp_dir, branch="master", depth=1)
            else:
                raise e
        
        # List of file extensions to include
        include_extensions = [
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.xml', '.html', '.css', '.sh', '.bash', '.sql', '.r', '.m'
        ]
        
        # Directories to exclude
        exclude_dirs = [
            '.git', 'node_modules', '__pycache__', '.pytest_cache',
            'venv', 'env', '.venv', 'dist', 'build', '.idea', '.vscode',
            'target', 'bin', 'obj', '.next', 'out', 'coverage'
        ]
        
        documents = []
        
        # Walk through the repository
        for root, dirs, files in os.walk(temp_dir):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                # Check if file has an included extension
                if any(file.endswith(ext) for ext in include_extensions):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, temp_dir)
                    
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Skip empty files or very large files (>1MB)
                        if content and len(content) < 1_000_000:
                            # Create document with metadata
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": relative_path,
                                    "file_name": file,
                                    "file_type": os.path.splitext(file)[1]
                                }
                            )
                            documents.append(doc)
                    
                    except Exception as e:
                        print(f"Error reading file {relative_path}: {e}")
                        continue
        
        print(f"Loaded {len(documents)} files from the repository")
        
        # Split documents into chunks (using larger chunks to reduce processing time)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased from 1000
            chunk_overlap=200,
            length_function=len,
        )
        
        split_documents = text_splitter.split_documents(documents)
        print(f"Split into {len(split_documents)} chunks")
        
        return split_documents
    
    finally:
        # Clean up temporary directory
        try:
            # On Windows, force removal with onerror handler
            def handle_remove_readonly(func, path, exc):
                import stat
                os.chmod(path, stat.S_IWRITE)
                func(path)
            
            shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
        except Exception as e:
            # Silently ignore cleanup errors as files are already processed
            pass

if __name__ == "__main__":
    # Test the loader
    test_repo = "https://github.com/streamlit/streamlit"
    docs = load_github_repo(test_repo)
    print(f"Total documents: {len(docs)}")
    if docs:
        print(f"Sample document: {docs[0].page_content[:200]}...")
