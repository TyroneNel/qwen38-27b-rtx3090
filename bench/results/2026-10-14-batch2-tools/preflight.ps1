# Batch 2 preflight: runs the brief's Step 0 commands plus the checks batch 1 kept, printing each command
# before its verbatim output. Batch 1's 3090 counter instance (luid_..114dd) is gone after a restart, so the
# counter step prints every adapter and the 3090 is identified from the output.
Set-Location G:\dev\HyperQwen
function run([string]$c) { "PS> $c"; Invoke-Expression $c 2>&1 | Out-String -Width 300; "" }
"# batch 2 preflight, {0}Z, PowerShell on the host" -f (Get-Date).ToUniversalTime().ToString('s')
""
run 'git fetch origin; git status -sb; git log --oneline -1'
run 'git rev-parse HEAD; git rev-parse --abbrev-ref HEAD'
run 'git config core.autocrlf'
run 'git config user.name; git config user.email'
run 'git remote -v'
run 'git config --get-all remote.upstream.pushurl; "exit=$LASTEXITCODE"'
run 'nvidia-smi --query-gpu=index,name,memory.total,power.limit,power.default_limit --format=csv'
run 'nvidia-smi --query-gpu=index,uuid,display_active,memory.used,memory.free,driver_version --format=csv'
run 'docker version --format "{{.Server.Version}}"'
run 'docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | Select-String hyperqwen'
run 'docker ps --format "{{.Names}} {{.Status}}"'
run 'Get-ChildItem G:\models | Select-Object Name'
run 'Get-PSDrive G | Select-Object Name,Used,Free'
run "Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 10 -MaxSamples 6 | ForEach-Object { `$_.CounterSamples | ForEach-Object { '{0} {1} {2:N2} MiB' -f `$_.Timestamp.ToString('s'), `$_.Path, (`$_.CookedValue/1MB) } }"
