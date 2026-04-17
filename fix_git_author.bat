@echo off
echo ====================================================
echo   MedRoute AI - Fixing Git Author Information
echo ====================================================
echo.

cd /d "%~dp0"

echo Setting Git config to katharv59...
git config user.name "katharv59"
git config user.email "katharv59email@gmail.com"

echo Resetting history to a single clean commit...
git checkout --orphan temp
git add -A
git commit -m "MedRoute AI v2.0.0 - Emergency Vehicle Priority System"

echo Replacing main branch...
git branch -D main
git branch -M main

echo Force pushing to GitHub...
git push -f origin main

echo.
echo ====================================================
echo   DONE! Refresh your GitHub page to see the changes.
echo ====================================================
pause
