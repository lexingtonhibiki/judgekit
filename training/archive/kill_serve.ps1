$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*serve_decisions.py*' }
if ($procs) {
  foreach ($p in $procs) {
    Write-Output ("killing PID " + $p.ProcessId + " : " + $p.Name)
    Stop-Process -Id $p.ProcessId -Force
  }
} else {
  Write-Output "no serve_decisions.py process found"
}
