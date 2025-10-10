# collect_changes.ps1
# Run from repository root. Creates ./changes_for_review with diffs, logs and a zip of changed files.

$repoRoot = (Get-Location).Path
$outDir = Join-Path $repoRoot "changes_for_review"

# create output dir
if (-Not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

Write-Host "Output directory: $outDir"

# Basic info
git rev-parse --abbrev-ref HEAD 2>$null | Out-File (Join-Path $outDir "current_branch.txt") -Encoding utf8
git status --porcelain | Out-File (Join-Path $outDir "git_status.txt") -Encoding utf8
git log --oneline -n 50 | Out-File (Join-Path $outDir "git_log_last_50.txt") -Encoding utf8

# Ensure we have latest remote refs for diffs
git fetch origin 2>$null

# Diffs and patches
# Diff between local HEAD and remote main (show which files changed and full diff)
git diff --name-only origin/main...HEAD | Out-File (Join-Path $outDir "changed_files_vs_origin_main.txt") -Encoding utf8
git diff origin/main...HEAD > (Join-Path $outDir "diff_vs_origin_main.patch") 2>$null
git format-patch origin/main --stdout > (Join-Path $outDir "commits_ahead.patch") 2>$null

# Uncommitted work
git diff > (Join-Path $outDir "workingtree_uncommitted.diff") 2>$null
git diff --staged > (Join-Path $outDir "staged.diff") 2>$null

# Untracked files
git ls-files --others --exclude-standard | Out-File (Join-Path $outDir "untracked_files.txt") -Encoding utf8

# Build a consolidated list of changed file paths to archive
$changed = @(git diff --name-only origin/main...HEAD 2>$null)
$unstaged = @(git status --porcelain 2>$null | ForEach-Object { if ($_.Length -ge 4) { $_.Substring(3) } })
$untracked = @(git ls-files --others --exclude-standard 2>$null)

$allFiles = @()
$allFiles += $changed
$allFiles += $unstaged
$allFiles += $untracked

# Keep only valid paths that exist in FS, remove duplicates
$allFiles = $allFiles | Where-Object { $_ -and (Test-Path $_) } | Sort-Object -Unique

if ($allFiles.Count -gt 0) {
    $zipPath = Join-Path $outDir "changed_files.zip"
    Compress-Archive -Path $allFiles -DestinationPath $zipPath -Force
    Write-Host "Created zip: $zipPath (`$allFiles.Count` files)"
} else {
    Write-Host "No changed file paths found to archive."
}

Write-Host "Saved status/logs/diffs to: $outDir"
