# Sustained timing: procedure

What it measures: the time of each pipeline stage on recorded video, with the frames released at the camera rate, the frames not finished when the next one arrives, the cost of the two IK solvers, and what the COMFI timestamp files say about the recordings. Results on the workstation: `timing-workstation.md`.

It does not measure the cameras' acquisition delay. Rates are processing rates on recorded video; a live system is capped by its cameras (40 Hz on COMFI).

## What is needed

- The Docker image of `main` and a clone of this branch.
- The models and engines. `bash scripts/bash/fetch_models.sh` downloads the models and builds the detector engines. Run it again with `nlf_backend = "tensorrt"` in `settings.py` for the NLF engine, and once more with `detection_mode = "tiled"` for the tiled detector engine; then put `settings.py` back. The IK solver is generated on the first run.
- The COMFI subset: participant 1012, tasks Screwing, Polishing, SideOverhead, RobotPolishing, RobotWelding, Lifting, with `videos/1012/<task>/camera_{0,2,4,6}.mp4`, `cam_params/1012` and `metadata/1012.yaml`.
- For the IK cost, the NLF output of the same trials, recorded once:
  `python3 scripts/python/paper/sweep.py --arm nlf --dataset <COMFI> --cameras 0 2 4 6 --participants 1012 --tasks Screwing Polishing SideOverhead RobotPolishing RobotWelding Lifting --tag nlf_0-2-4-6 --output-dir <runs> --summary <runs>/nlf.csv --views-cache <views>`
- An idle machine: no other job on the GPU or the CPU. The report states the load average at start.

## Commands

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/python/paper/sustained_timing.py replay     --dataset <COMFI> --out results/timing
python3 scripts/python/paper/sustained_timing.py ik         --dataset <COMFI> --out results/timing --views <views>
python3 scripts/python/paper/sustained_timing.py timestamps --dataset <COMFI> --out results/timing
python3 scripts/python/paper/sustained_timing.py report     --dataset <COMFI> --out results/timing
```

`replay` takes about seven minutes per inference configuration (three: TorchScript, TensorRT, TensorRT with tiled detection). `timestamps` needs the videos' timestamp files of the 18 participants and can be skipped on a subset. The OpenCap pass (`opencap`, with `--dataset <LabValidation_withVideos>`) is optional: subject 4, four cameras at 60 Hz.

## Reading the report

- Stages: `decode` is the wait for the next decoded frames (the reader decodes ahead), `upload` the copy to the GPU, `crop` and `nlf` the two parts of NLF (TorchScript does both in `nlf`), `output` the writing of markers and angles. Each stage waits for the GPU; with `rest of the inference` (reading NLF's output back) the stages add up to the processing time.
- A frame misses its deadline when its joint angles are not ready when the next frame arrives; a late frame delays the ones behind it until the backlog clears.
- Latency of a motion event is the wait for the next frame (spread over one period), the delay from the frame's arrival to its joint angles, and the lag of the estimate behind the reference. The delay is computed from the measured durations of frames arriving every period.
- The timestamps are arrival times at the recording computer: they bound dropped frames and give the spread between cameras, not exposure times.
- The first 40 frames of each trial are left out (model calibration).
