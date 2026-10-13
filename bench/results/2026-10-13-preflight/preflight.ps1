Set-Location G:\dev\HyperQwen
function Run($c) { "PS> $c"; Invoke-Expression $c 2>&1 | Out-String -Width 400; "" }
Run 'git status --short'
Run 'git log --oneline -1'
Run 'git rev-parse --abbrev-ref HEAD'
Run 'git config core.autocrlf'
Run 'git config --show-origin --get-all core.autocrlf'
Run 'nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,power.default_limit,power.max_limit,power.draw,display_active,display_mode,pci.bus_id,uuid --format=csv'
Run 'nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,power.draw --format=csv'
Run "docker version --format '{{.Server.Version}}'"
Run "docker info --format '{{.OSType}}'"
Run "docker info --format '{{.DockerRootDir}} {{.OperatingSystem}} {{.MemTotal}} {{.NCPU}}'"
Run 'docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi -L'
Run 'Get-ChildItem G:\models | Select-Object Name'
Run 'Get-PSDrive -PSProvider FileSystem | Select-Object Name,@{n="FreeGB";e={[math]::Round($_.Free/1GB,1)}},@{n="UsedGB";e={[math]::Round($_.Used/1GB,1)}}'
Run 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
Run 'nvidia-smi'
Run "Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 10 -MaxSamples 6"
Run 'nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv'
