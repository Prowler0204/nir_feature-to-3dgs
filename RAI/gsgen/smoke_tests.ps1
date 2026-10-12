$ErrorActionPreference = "Continue"

$env:WANDB_MODE = "disabled"
$env:WANDB_SILENT = "true"
$env:PYTHONUNBUFFERED = "1"

$MAX_STEPS = 500
$RESO = "[128]"
$CONFIG = "base"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$SUMMARY = "logs\_summary.txt"
Set-Content -Path $SUMMARY -Value ""

function Run-Test {
    param(
        [string]$tag,
        [string]$prompt
    )
    Write-Host ""
    Write-Host "=== [$tag] $prompt ==="
    
    $log_file = "logs\${tag}.log"
    
    # Run the model (assumes you are already in the gsgen env), explicitly disabling wandb via hydra override
    python main.py --config-name=$CONFIG prompt.prompt="$prompt" max_steps=$MAX_STEPS data.reso=$RESO wandb=false 2>&1 | Tee-Object -FilePath $log_file
    
    if ($LASTEXITCODE -eq 0) {
        Add-Content -Path $SUMMARY -Value "${tag}: OK"
        Write-Host "${tag}: OK"
    } else {
        Add-Content -Path $SUMMARY -Value "${tag}: FAILED (see logs\${tag}.log)"
        Write-Host "${tag}: FAILED (see logs\${tag}.log)"
    }
}

Run-Test "0_control_en" "A simple wooden dining table with four chairs."
Run-Test "0_control_hi" "Ek simple wooden dining table chaar chairs ke saath."

Run-Test "1_temple_en" "A highly detailed traditional Hindu temple made of carved stone."
Run-Test "1_temple_hi" "Ek highly detailed stone ka mandir."

Run-Test "2_kulhad_en" "A hot cup of Indian spiced tea served in a traditional terracotta clay cup."
Run-Test "2_kulhad_hi" "Mitti ke kulhad mein garam masala chai."

Run-Test "3_rickshaw_en" "A brightly decorated three-wheeler auto rickshaw."
Run-Test "3_rickshaw_hi" "Ek brightly decorated auto rickshaw."

Write-Host ""
Write-Host "=== SMOKE TEST SUMMARY ==="
Get-Content $SUMMARY
