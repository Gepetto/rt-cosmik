# Quick start

No cameras needed: this runs RT-COSMIK on a 20-second recording from the
[COMFI dataset](https://doi.org/10.5281/zenodo.17223909), someone welding with a
Franka robot, filmed by four cameras.

You need Linux, an NVIDIA GPU, [Docker](https://docs.docker.com/engine/install/)
and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The Docker image holds everything else. To install without Docker, see
[Installation](installation.md).

## 1. Get the code and start the container

```bash
git clone https://github.com/Gepetto/rt-cosmik.git
cd rt-cosmik
docker/run.sh
```

The first run builds the image, which takes about half an hour: it compiles
CasADi, Pinocchio and acados. After that, `docker/run.sh` opens a shell in the
container within seconds, with your checkout at `/root/workspace/rt-cosmik`. VS
Code users can use **Dev Containers: Reopen in Container** instead.

## 2. Download the models and generate the solver

Once, inside the container:

```bash
# the pose estimator and the person detector
scripts/bash/fetch_models.sh
# the inverse kinematics solver
python3 scripts/python/core/run_ocp_codegen.py --backend acados --profile realtime
```

The first downloads the network weights and builds a TensorRT engine of the
detector for each camera count. The second generates and compiles the solver,
in about a minute.

## 3. Run the sample

```bash
scripts/bash/fetch_sample.sh
python3 scripts/python/core/run_pipeline.py \
    --dataset data/comfi_sample --participant 2112 --task RobotWelding
```

Open the URL it prints, <http://127.0.0.1:7000/static/>, to watch the
reconstruction in 3D: the body, the cameras, the table, and the robot moving as
it was recorded. The results are written to
`output/2112/RobotWelding/4cam_mhe_acados/`:

| File | Contents |
|---|---|
| `joint_angles.csv` | One row per frame: pelvis position (m) and orientation (quaternion), then the 36 joint angles (rad), with names such as `Right_Knee_Flexion_Extension[rad]` |
| `markers.csv` | The fused 3D landmarks (m), in the world frame |
| `run_info.json` | The configuration of the run and the person's calibrated model |

Their columns, units and frames are described in
[Outputs and the human model](outputs.md).

## 4. Compare with motion capture

The sample includes the marker-based reference of the same trial:

```bash
python3 scripts/python/eval/compare_to_mocap.py \
    --reference data/comfi_sample/mocap/aligned/2112/RobotWelding \
    rt-cosmik=output/2112/RobotWelding/4cam_mhe_acados \
    --plots output/2112/eval
```

It prints the error of every joint angle and marker (about 9° and 45 mm on
average on this trial) and draws them in `output/2112/eval/`. Comparisons of
several runs, and processing of whole datasets, are covered in
[Recorded data and evaluation](offline.md).

## Next

- [Your own cameras](own-cameras.md): calibrate a rig and run live.
- [Overview](overview.md): what each stage of the pipeline does.
