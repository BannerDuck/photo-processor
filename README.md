# Photo Processor

This Dockerized Python application watches a folder for new images, rotates, crops, and removes the background and sets the background to white using Mediapipe and Rembg.

## How to Use

1. Edit `config.yaml` to set your folder paths for:
   - `temp_dir`
   - `processed_dir`
   - `processed_dir_rembg`
   - `originals_dir`
   - `log_file`

2. Build and run with Docker Compose:

```bash
docker-compose up -d --build
