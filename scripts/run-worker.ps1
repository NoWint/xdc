param([string]$Config = ".\xdc-demo\worker1.yaml")
& .\.venv\Scripts\xdc.exe worker --config $Config
