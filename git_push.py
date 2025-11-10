"""
🚀 Git Push Automation Script
=============================
Automates pushing to GitHub 'prediction' branch

Usage:
    python git_push.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command safely and handle errors"""
    print(f"\n{'='*70}")
    if description:
        print(f"📌 {description}")
    print(f"{'='*70}")
    print(f"Running: {command}\n")
    
    try:
        # ✅ Force Windows to use cmd.exe for Git commands
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            executable="C:\\Windows\\System32\\cmd.exe"
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        print("✅ Success!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except OSError as e:
        print(f"❌ OS Error: {e}")
        print("💡 Tip: Make sure Git is installed and in your PATH.")
        return False


def create_gitignore():
    """Create or overwrite .gitignore file"""
    print("\n" + "="*70)
    print("📝 Creating .gitignore file")
    print("="*70)
    
    gitignore_content = """__pycache__/
*.py[cod]
.Python
venv/
env/
.vscode/
.idea/
.DS_Store
*.bin
*.safetensors
*.log
cache/*.faiss
cache/*.npy
!cache/metadata.pkl
!cache/model_info.json
!cache/cross_store_synonyms.pkl
"""
    
    try:
        with open('.gitignore', 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("✅ .gitignore created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating .gitignore: {e}")
        return False


def main():
    """Main automation function"""
    print("\n" + "="*70)
    print("🚀 GIT PUSH AUTOMATION - Branch: prediction")
    print("="*70)
    print("Repository: https://github.com/zenmeraki7/assureful_categories")
    print("="*70)
    
    # Step 1: Create .gitignore
    create_gitignore()
    
    # Step 2: Check Git status
    if not run_command("git status", "Checking Git status"):
        print("\n❌ Not a git repository. Run: git init")
        sys.exit(1)
    
    # Step 3: Ensure we're on 'prediction' branch
    print("\n" + "="*70)
    print("🌿 Ensuring 'prediction' branch exists")
    print("="*70)
    
    result = subprocess.run(
        "git branch --list prediction",
        shell=True,
        capture_output=True,
        text=True,
        executable="C:\\Windows\\System32\\cmd.exe"
    )
    
    if "prediction" in result.stdout:
        print("Branch 'prediction' already exists — switching to it...")
        run_command("git checkout prediction", "Switching to 'prediction' branch")
    else:
        print("Creating new 'prediction' branch...")
        run_command("git checkout -b prediction", "Creating 'prediction' branch")
    
    # Step 4: Stage all changes
    if not run_command("git add .", "Adding all files to staging"):
        print("\n❌ Failed to add files")
        sys.exit(1)
    
    # Step 5: Commit
    commit_message = """feat: Production API server with E5-Base model

- Flask API server with GPU acceleration
- Batch classification endpoint (/api/batch)
- 33,303 categories indexed
- AI-generated synonyms (1,000 mappings)
- Web UI with real-time classification
- Health check endpoint
- E5-Base (768D) embeddings
- Windows + NVIDIA GPU optimized"""
    
    print("\n" + "="*70)
    print("💾 Committing changes")
    print("="*70)
    print(f"Message:\n{commit_message}\n")
    
    result = subprocess.run(
        f'git commit -m "{commit_message}"',
        shell=True,
        capture_output=True,
        text=True,
        executable="C:\\Windows\\System32\\cmd.exe"
    )
    
    if result.returncode != 0:
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            print("ℹ️ Nothing to commit (working tree clean)")
        else:
            print(f"❌ Commit failed: {result.stderr}")
            sys.exit(1)
    else:
        print("✅ Committed successfully!")
    
    # Step 6: Remote repo check
    print("\n" + "="*70)
    print("🔗 Checking remote repository")
    print("="*70)
    
    result = subprocess.run(
        "git remote -v",
        shell=True,
        capture_output=True,
        text=True,
        executable="C:\\Windows\\System32\\cmd.exe"
    )
    print(result.stdout)
    
    if "origin" not in result.stdout:
        print("\n⚠️ No 'origin' remote found — adding it...")
        run_command(
            "git remote add origin https://github.com/zenmeraki7/assureful_categories.git",
            "Adding remote origin"
        )
    
    # Step 7: Push to GitHub
    print("\n" + "="*70)
    print("🚀 PUSHING TO GITHUB")
    print("="*70)
    print("Target: origin/prediction\n")
    
    if not run_command("git push -u origin prediction", "Pushing to GitHub"):
        print("\n❌ Push failed!")
        print("\nPossible issues:")
        print("1️⃣ Authentication required — login to GitHub")
        print("2️⃣ Remote repo missing")
        print("3️⃣ Large files (>100MB) in repo")
        print("\nTry manually: git push -u origin prediction")
        sys.exit(1)
    
    # Success
    print("\n" + "="*70)
    print("🎉 SUCCESS! CODE PUSHED TO GITHUB")
    print("="*70)
    print("\n📍 View your branch:")
    print("   https://github.com/zenmeraki7/assureful_categories/tree/prediction")
    print("\n📍 Create Pull Request:")
    print("   https://github.com/zenmeraki7/assureful_categories/pull/new/prediction")
    print("\n✅ All done!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
