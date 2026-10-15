# Batch 4 Step 0: the brief's preflight commands, verbatim where possible, output to stdout.
Set-Location G:\dev\HyperQwen
"# preflight " + (Get-Date).ToUniversalTime().ToString('s') + "Z"
"> git fetch origin; git status -sb; git log --oneline -1"
git fetch origin 2>&1; git status -sb 2>&1; git log --oneline -1
"> nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv
"> nvidia-smi --query-gpu=uuid,display_active --format=csv"
nvidia-smi --query-gpu=uuid,display_active --format=csv
"> docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | Select-String hyperqwen"
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | Select-String hyperqwen
"> docker ps -a"
docker ps -a --format '{{.Names}} {{.Status}}'
"> Get-ChildItem G:\models | Select-Object Name"
Get-ChildItem G:\models | Select-Object -ExpandProperty Name
"> Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 10 -MaxSamples 6"
$s = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 10 -MaxSamples 6
foreach ($x in $s) { foreach ($c in $x.CounterSamples) { "{0} {1} {2:N2} MiB" -f $x.Timestamp.ToString('s'), $c.Path, ($c.CookedValue / 1MB) } }
