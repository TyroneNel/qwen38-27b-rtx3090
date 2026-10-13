# Samples the Windows GPU adapter memory counters every 10 s until $Stop exists.
# Output CSV: timestamp,instance,counter,bytes. The 3090 is luid_0x00000000_0x000114dd_phys_0 (confirmed at the E1-A boot).
param([string]$Out, [string]$Stop)
"timestamp,instance,counter,bytes" | Set-Content -Encoding ascii $Out
while (-not (Test-Path $Stop)) {
  $s = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage'
  foreach ($c in $s.CounterSamples) {
    $parts = $c.Path -split '\\'
    $inst = ($parts[3] -replace '^gpu adapter memory\(', '') -replace '\)$', ''
    "{0},{1},{2},{3}" -f $s.Timestamp.ToString('s'), $inst, $parts[4], [int64]$c.CookedValue | Add-Content -Encoding ascii $Out
  }
  Start-Sleep -Seconds 9
}
