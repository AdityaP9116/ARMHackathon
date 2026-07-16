# sync.ps1 — one-command GitHub sync for the ARMHackathon repo
#
# Usage:
#   .\sync.ps1              # pull latest from GitHub (run at the START of a session)
#   .\sync.ps1 -m "message" # commit everything and push (run at the END of a session)
#
# Run it from the project folder in PowerShell.

param([string]$m = "")

Write-Host "Pulling latest from GitHub..." -ForegroundColor Cyan
git pull --rebase

if ($m -ne "") {
    Write-Host "Committing and pushing your changes..." -ForegroundColor Cyan
    git add -A
    git commit -m $m
    git push
    Write-Host "Done — changes are live on GitHub." -ForegroundColor Green
} else {
    Write-Host "Up to date. (Pass -m ""your message"" to commit + push.)" -ForegroundColor Green
}
