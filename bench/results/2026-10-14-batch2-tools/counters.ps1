# One snapshot of every GPU adapter's Dedicated and Shared Usage, in MiB, to identify the 3090's instance.
(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage').CounterSamples |
  ForEach-Object { '{0} {1:N2} MiB' -f $_.Path, ($_.CookedValue / 1MB) }
