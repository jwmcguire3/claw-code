$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$savedLocationsFile = Join-Path $scriptRoot "saved-locations.json"
$optionsFile = Join-Path $scriptRoot "launcher-options.json"

function Write-Utf8NoBom {
    param(
        [string]$Path,
        [string]$Content
    )
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

function Load-JsonFile {
    param(
        [string]$Path,
        $DefaultValue
    )

    if (-not (Test-Path $Path)) {
        $json = $DefaultValue | ConvertTo-Json -Depth 10
        Write-Utf8NoBom -Path $Path -Content $json
        return $DefaultValue
    }

    $raw = [System.IO.File]::ReadAllText($Path)
    if ([string]::IsNullOrWhiteSpace($raw)) {
        $json = $DefaultValue | ConvertTo-Json -Depth 10
        Write-Utf8NoBom -Path $Path -Content $json
        return $DefaultValue
    }

    return $raw | ConvertFrom-Json
}

function Save-JsonFile {
    param(
        [string]$Path,
        $Value
    )
    $json = $Value | ConvertTo-Json -Depth 10
    Write-Utf8NoBom -Path $Path -Content $json
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Pause-IfNeeded {
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Confirm-YesNo {
    param(
        [string]$Prompt,
        [bool]$Default = $false
    )

    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }

    while ($true) {
        $value = Read-Host "$Prompt $suffix"
        if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
        if ($value -match '^[Yy]$') { return $true }
        if ($value -match '^[Nn]$') { return $false }
        Write-Host "Enter y or n." -ForegroundColor Yellow
    }
}

function Select-FromList {
    param(
        [string]$Title,
        [array]$Items,
        [string]$NewLabel = "",
        [switch]$AllowBack,
        [switch]$AllowCustom,
        [string]$DefaultValue = ""
    )

    while ($true) {
        Write-Host ""
        Write-Host $Title
        Write-Host ("-" * $Title.Length)

        for ($i = 0; $i -lt $Items.Count; $i++) {
            $item = [string]$Items[$i]
            $marker = if ($item -eq $DefaultValue) { " (default)" } else { "" }
            Write-Host ("[{0}] {1}{2}" -f ($i + 1), $item, $marker)
        }

        if ($NewLabel) {
            Write-Host "[N] $NewLabel"
        }
        if ($AllowCustom) {
            Write-Host "[C] Custom value"
        }
        if ($AllowBack) {
            Write-Host "[B] Back"
        }
        if (-not [string]::IsNullOrWhiteSpace($DefaultValue)) {
            Write-Host "[Enter] Use default: $DefaultValue"
        }
        Write-Host "[Q] Quit"

        $choice = Read-Host "Select an option"

        if ([string]::IsNullOrWhiteSpace($choice) -and -not [string]::IsNullOrWhiteSpace($DefaultValue)) {
            return @{ Type = "default"; Value = $DefaultValue }
        }
        if ($choice -match '^[Qq]$') { return @{ Type = "quit"; Value = $null } }
        if ($AllowBack -and $choice -match '^[Bb]$') { return @{ Type = "back"; Value = $null } }
        if ($NewLabel -and $choice -match '^[Nn]$') { return @{ Type = "new"; Value = $null } }
        if ($AllowCustom -and $choice -match '^[Cc]$') { return @{ Type = "custom"; Value = $null } }

        $index = 0
        if ([int]::TryParse($choice, [ref]$index)) {
            if ($index -ge 1 -and $index -le $Items.Count) {
                return @{ Type = "item"; Value = $Items[$index - 1] }
            }
        }

        Write-Host "Invalid choice." -ForegroundColor Yellow
    }
}

function Get-SavedLocations {
    $locations = Load-JsonFile -Path $savedLocationsFile -DefaultValue @()
    if ($locations -is [string]) { return @($locations) }
    return @($locations)
}

function Save-SavedLocations {
    param([array]$Locations)
    $clean = @($Locations | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
    Save-JsonFile -Path $savedLocationsFile -Value $clean
}

function Select-BaseLocation {
    while ($true) {
        $locations = Get-SavedLocations
        $selection = Select-FromList -Title "Saved base locations" -Items $locations -NewLabel "Add a new base location"
        switch ($selection.Type) {
            "quit" { return $null }
            "new" {
                $newBase = Read-Host "Enter a base folder path"
                if (-not [string]::IsNullOrWhiteSpace($newBase)) {
                    Ensure-Directory -Path $newBase
                    $locations += $newBase
                    Save-SavedLocations -Locations $locations
                    return $newBase
                }
            }
            "item" { return [string]$selection.Value }
            "default" { return [string]$selection.Value }
        }
    }
}

function Select-WorkspaceFolder {
    param([string]$BasePath)

    Ensure-Directory -Path $BasePath

    while ($true) {
        $subfolders = @(Get-ChildItem -Path $BasePath -Directory -Force -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -ExpandProperty FullName)
        $selection = Select-FromList -Title "Folders inside $BasePath" -Items $subfolders -NewLabel "Create a new repo folder here" -AllowBack

        switch ($selection.Type) {
            "quit" { return @{ Type = "quit"; Value = $null } }
            "back" { return @{ Type = "back"; Value = $null } }
            "item" { return @{ Type = "existing"; Value = [string]$selection.Value } }
            "new" {
                $repoName = Read-Host "Enter the new folder name"
                if ([string]::IsNullOrWhiteSpace($repoName)) {
                    Write-Host "Folder name cannot be empty." -ForegroundColor Yellow
                    continue
                }
                $repoPath = Join-Path $BasePath $repoName
                Ensure-Directory -Path $repoPath
                return @{ Type = "new"; Value = $repoPath }
            }
        }
    }
}

function Ensure-GitRepo {
    param([string]$RepoPath)

    $gitPath = Join-Path $RepoPath ".git"
    if (-not (Test-Path $gitPath)) {
        Write-Host ""
        Write-Host "Initializing Git repository..." -ForegroundColor Cyan
        Push-Location $RepoPath
        try {
            git init | Out-Host
            git branch -M main 2>$null
        }
        finally {
            Pop-Location
        }
    }
}

function New-Readme {
    param([string]$RepoPath)
    $readmePath = Join-Path $RepoPath "README.md"
    if (-not (Test-Path $readmePath)) {
        $content = @"
# $(Split-Path $RepoPath -Leaf)

## Purpose

## Notes

## Run

## Todo
"@
        Write-Utf8NoBom -Path $readmePath -Content $content
    }
}

function New-Gitignore {
    param([string]$RepoPath)
    $path = Join-Path $RepoPath ".gitignore"
    if (-not (Test-Path $path)) {
        $content = @"
.claw/
node_modules/
dist/
build/
.env
.env.local
*.log
"@
        Write-Utf8NoBom -Path $path -Content $content
    }
}

function New-ClawSettings {
    param(
        [string]$RepoPath,
        [string]$DefaultModel
    )

    $clawDir = Join-Path $RepoPath ".claw"
    $settingsPath = Join-Path $clawDir "settings.json"
    Ensure-Directory -Path $clawDir

    if (-not (Test-Path $settingsPath)) {
        $settings = @{
            aliases = @{
                fast = $DefaultModel
            }
        }
        Save-JsonFile -Path $settingsPath -Value $settings
    }
}

function Initialize-NewRepoFiles {
    param(
        [string]$RepoPath,
        $Options
    )

    if ($Options.newRepoTemplates.createReadme) {
        New-Readme -RepoPath $RepoPath
    }
    if ($Options.newRepoTemplates.createGitignore) {
        New-Gitignore -RepoPath $RepoPath
    }
    if ($Options.newRepoTemplates.createClawSettings) {
        New-ClawSettings -RepoPath $RepoPath -DefaultModel ([string]$Options.defaultModel)
    }
}

function Select-Model {
    param($Options)
    $selection = Select-FromList -Title "Select a model" -Items @($Options.models) -AllowCustom -DefaultValue ([string]$Options.defaultModel)
    switch ($selection.Type) {
        "quit" { return $null }
        "custom" {
            $custom = Read-Host "Enter custom model string"
            if ([string]::IsNullOrWhiteSpace($custom)) { return [string]$Options.defaultModel }
            return $custom
        }
        "item" { return [string]$selection.Value }
        "default" { return [string]$selection.Value }
        default { return [string]$Options.defaultModel }
    }
}

function Select-PermissionMode {
    param($Options)
    $selection = Select-FromList -Title "Select permission mode" -Items @($Options.permissionModes) -DefaultValue ([string]$Options.defaultPermissionMode)
    if ($selection.Type -eq "quit") { return $null }
    return [string]$selection.Value
}

function Select-OutputFormat {
    param($Options)
    $selection = Select-FromList -Title "Select output format" -Items @($Options.outputFormats) -DefaultValue ([string]$Options.defaultOutputFormat)
    if ($selection.Type -eq "quit") { return $null }
    return [string]$selection.Value
}

function Select-AllowedTools {
    param($Options)
    $selection = Select-FromList -Title "Select allowedTools preset" -Items @($Options.commonAllowedToolsSets) -AllowCustom -DefaultValue ([string]$Options.defaultAllowedTools)
    switch ($selection.Type) {
        "quit" { return $null }
        "custom" {
            $custom = Read-Host "Enter allowedTools value (comma-separated)"
            if ([string]::IsNullOrWhiteSpace($custom)) { return [string]$Options.defaultAllowedTools }
            return $custom
        }
        "item" { return [string]$selection.Value }
        "default" { return [string]$selection.Value }
        default { return [string]$Options.defaultAllowedTools }
    }
}

function Select-LaunchMode {
    param($Options)
    $items = @("interactive", "one-shot prompt")
    $selection = Select-FromList -Title "Select launch mode" -Items $items -DefaultValue ([string]$Options.defaultLaunchMode)
    if ($selection.Type -eq "quit") { return $null }
    return [string]$selection.Value
}

function Build-LaunchConfig {
    param($Options)

    $model = Select-Model -Options $Options
    if ($null -eq $model) { return $null }

    $permissionMode = Select-PermissionMode -Options $Options
    if ($null -eq $permissionMode) { return $null }

    $outputFormat = Select-OutputFormat -Options $Options
    if ($null -eq $outputFormat) { return $null }

    $allowedTools = Select-AllowedTools -Options $Options
    if ($null -eq $allowedTools) { return $null }

    $resumeLatest = Confirm-YesNo -Prompt "Resume latest session?" -Default ([bool]$Options.defaultResumeLatest)
    $launchMode = Select-LaunchMode -Options $Options
    if ($null -eq $launchMode) { return $null }

    $oneShotPrompt = $null
    if ($launchMode -eq "one-shot prompt") {
        $oneShotPrompt = Read-Host "Enter prompt text"
        if ([string]::IsNullOrWhiteSpace($oneShotPrompt)) {
            Write-Host "Prompt cannot be empty for one-shot mode." -ForegroundColor Yellow
            return $null
        }
    }

    return [ordered]@{
        model = $model
        permissionMode = $permissionMode
        outputFormat = $outputFormat
        allowedTools = $allowedTools
        resumeLatest = $resumeLatest
        launchMode = $launchMode
        oneShotPrompt = $oneShotPrompt
    }
}

function Show-LaunchSummary {
    param(
        [string]$Workspace,
        $Config,
        [string]$ClawExe
    )

    $parts = @("& `"$ClawExe`"")
    $parts += "--model `"$($Config.model)`""
    $parts += "--permission-mode `"$($Config.permissionMode)`""
    $parts += "--output-format `"$($Config.outputFormat)`""

    if (-not [string]::IsNullOrWhiteSpace($Config.allowedTools)) {
        $parts += "--allowedTools `"$($Config.allowedTools)`""
    }

    if ($Config.resumeLatest) {
        $parts += "--resume latest"
    }

    if ($Config.launchMode -eq "one-shot prompt") {
        $escapedPrompt = $Config.oneShotPrompt.Replace('"', '\"')
        $parts += "prompt `"$escapedPrompt`""
    }

    Write-Host ""
    Write-Host "Workspace: $Workspace" -ForegroundColor Cyan
    Write-Host "Command:" -ForegroundColor Cyan
    Write-Host ($parts -join " ")
}

function Launch-Claw {
    param(
        [string]$Workspace,
        [string]$ClawExe,
        $Config
    )

    if (-not (Test-Path $ClawExe)) {
        throw "claw.exe not found at: $ClawExe"
    }

    $env:HOME = $env:USERPROFILE

    Push-Location $Workspace
    try {
        $args = @(
            "--model", $Config.model,
            "--permission-mode", $Config.permissionMode,
            "--output-format", $Config.outputFormat
        )

        if (-not [string]::IsNullOrWhiteSpace($Config.allowedTools)) {
            $args += @("--allowedTools", $Config.allowedTools)
        }

        if ($Config.resumeLatest) {
            $args += @("--resume", "latest")
        }

        if ($Config.launchMode -eq "one-shot prompt") {
            $args += @("prompt", $Config.oneShotPrompt)
        }

        & $ClawExe @args
        Write-Host ""
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor DarkGray
    }
    finally {
        Pop-Location
    }
}

$options = Load-JsonFile -Path $optionsFile -DefaultValue @{
    clawExe = "C:\Users\jwmcg\OneDrive\Documents\AI Projects\Claw Code\rust\target\debug\claw.exe"
    defaultModel = "openai/gpt-4.1-mini"
    defaultPermissionMode = "workspace-write"
    defaultOutputFormat = "text"
    defaultAllowedTools = "read,glob,grep,write,edit"
    defaultResumeLatest = $false
    defaultLaunchMode = "interactive"
    models = @(
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1",
        "deepseek-chat",
        "deepseek-reasoner",
        "opus",
        "sonnet",
        "haiku",
        "grok",
        "grok-mini",
        "qwen-plus"
    )
    permissionModes = @("read-only", "workspace-write", "danger-full-access")
    outputFormats = @("text", "json")
    commonAllowedToolsSets = @("", "read,glob", "read,glob,grep", "read,glob,grep,write,edit")
    newRepoTemplates = @{
        createReadme = $true
        createGitignore = $true
        createClawSettings = $true
    }
}

while ($true) {
    $base = Select-BaseLocation
    if ($null -eq $base) { break }

    $workspaceSelection = Select-WorkspaceFolder -BasePath $base
    if ($workspaceSelection.Type -eq "quit") { break }
    if ($workspaceSelection.Type -eq "back") { continue }

    $workspace = [string]$workspaceSelection.Value

    if ($workspaceSelection.Type -eq "new") {
        Ensure-GitRepo -RepoPath $workspace
        Initialize-NewRepoFiles -RepoPath $workspace -Options $options
    }

    $launchConfig = Build-LaunchConfig -Options $options
    if ($null -eq $launchConfig) { continue }

    Show-LaunchSummary -Workspace $workspace -Config $launchConfig -ClawExe ([string]$options.clawExe)
    $go = Confirm-YesNo -Prompt "Launch now?" -Default $true
    if (-not $go) { continue }

    Launch-Claw -Workspace $workspace -ClawExe ([string]$options.clawExe) -Config $launchConfig
    Pause-IfNeeded
    break
}
