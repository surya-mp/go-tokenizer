# Test Report

Date: 2026-09-12
Host: Windows, PowerShell
GPU: NVIDIA GeForce RTX 3050 Laptop GPU detected by `nvidia-smi`; CUDA version reported by driver: 13.1.

## Commands Run

```powershell
$env:GOCACHE=(Join-Path (Get-Location) '.gocache')
$env:GOMODCACHE=(Join-Path (Get-Location) '.gomodcache')
go test ./...
```

## Result

Pass.

Validated package:

- `github.com/surya-mp/go-tokenizer`

## CUDA/GPU Notes

This repository is tokenizer-focused and does not contain CUDA code or GPU tests, so no GPU execution was expected or performed.
