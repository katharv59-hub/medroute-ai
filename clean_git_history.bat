@echo off
title Clean Git History - MedRoute AI
echo ============================================
echo   ASEP — Securing Git History
echo ============================================
echo.
echo This script will remove tracked sensitive JSON keys from your local Git history.
echo It uses 'git rm --cached' so your local files won't be deleted, but they will
echo no longer be tracked by Git.
echo.

git rm --cached "serviceAccountKey.json" 2>nul
git rm --cached "ev-priority-sys-001-d6a3a0546f1e.json" 2>nul

echo.
echo Adding tracked files to the commit to finalize their removal from the index...
git add .gitignore
git commit -m "chore: remove exposed credentials and update gitignore"

echo.
echo ============================================
echo   DONE!
echo ============================================
echo Make sure to run 'git push origin main' to apply these changes to GitHub.
echo If the files were pushed in old commits, they might still be visible in the
echo history on GitHub. For complete removal from history, you should use tools
echo like BFG Repo-Cleaner or 'git filter-repo'.
echo.
pause
