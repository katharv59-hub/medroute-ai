# MedRoute AI Security Report

## 1. Secrets Found & Removed
- Tracked Firebase service account JSON files were identified in the codebase:
  - `serviceAccountKey.json`
  - `ev-priority-sys-001-d6a3a0546f1e.json`
- These files contained the private key for the Firebase Admin SDK, which provides full read/write access to the database.

## 2. Actions Taken to Secure the Repository
- **Git History Cleaning Script**: Created `clean_git_history.bat` for you to run locally, which untracks the compromised secrets from Git.
- **Environment Variables Support**: Implemented `python-dotenv` to securely load the Firebase configuration from a local `.env` file rather than hardcoding the file path.
- **`.env.example` Template**: Created a safe template showing how to set `FIREBASE_KEY_PATH` without storing actual credentials in the repository.
- **`.gitignore` Hardened**: Updated the ignore file to explicitly block `serviceAccountKey.json`, `ev-priority-sys-001-*.json`, `.env`, and backup files `*.bak`.
- **Graceful Error Handling**:
  - Modified `firebase_sender.py` to fail safely and disable Firebase features if the key is missing. The system will no longer crash or print the absolute path of the secret key to the console.
  - Added runtime validation to `detect.py` to check for the existence of the Firebase key path configured in `.env`, the model file, and the video source before execution.
- **Dashboard Annotations**: Verified that `dashboard/index.html` only contains public Firebase client keys. Added explicit warning comments advising against pasting private service account credentials in the HTML.

## 3. Remaining Risks & Recommendations
1. **Rotate the Exposed Key**: Because the keys were previously tracked in Git, you should **delete the exposed service account key** from the Firebase Console (Project Settings -> Service Accounts) and generate a new one if you haven't already. The previous key is compromised if the repository was ever pushed to a public location.
2. **Push the Cleaned History**: After running `clean_git_history.bat`, you must run `git push origin main` to apply the untracking to GitHub. If the secret was pushed in past commits, you should use BFG Repo-Cleaner or `git filter-repo` to permanently erase the files from your GitHub history.
3. **Local Setup Required**: When cloning this project elsewhere, ensure you create the `.env` file manually and place your new `serviceAccountKey.json` safely in the root folder, as these files will not be included in Git.
