param([string]$Config = ".\xdc-demo\coordinator.yaml")
& .\.venv\Scripts\xdc.exe coordinator --config $Config --access-log
