# Formula Edge Macau 2024 · Autonomous JetRacer

University Division · 1st Runner-up · ZENITH RACING · Macau University of Science and Technology

An archive of our entry in the Inter-School A.I. Formula Edge Competition, Macau 2024: a Jetson Nano racecar, camera-based road following, printable hardware files, and footage from the competition.

本仓库记录 ZENITH RACING 参加 2024 年澳门站 Formula Edge 比赛并获得 JetRacer 大学组亚军的经历，整理了现场影像、硬件打印文件、比赛资料，以及 NVIDIA JetRacer 官方代码的固定版本。

[Race footage](#race-footage) · [Technical workflow](#technical-workflow) · [Getting started](#getting-started) · [Hardware](#hardware-and-3d-printing) · [Repository contents](#repository-contents)

<table>
  <tr>
    <td width="50%"><img src="assets/raceday_images/IMG_5831.jpg" alt="JetRacer competition car with Jetson Nano, camera, radio controller, and trophy" width="460"></td>
    <td width="50%"><img src="assets/raceday_images/IMG_5842.jpg" alt="Formula Edge Macau 2024 university final first runner-up trophy" width="460"></td>
  </tr>
  <tr>
    <td align="center">Competition car and equipment</td>
    <td align="center">University final · 1st Runner-up</td>
  </tr>
</table>

## Competition result

| Item | Details |
| --- | --- |
| Event | Inter-School A.I. Formula Edge Competition, Macau 2024 |
| Category | JetRacer · University Division |
| Team | ZENITH RACING (UR04) |
| Institution | Macau University of Science and Technology (澳門科技大學) |
| Result | 1st Runner-up / Second Place |

The archived [final-results graphic](assets/raceday_images/race_result.avif) lists UR04 ZENITH RACING as the first runner-up. The trophy photograph above provides a second record of the award.

<details>
<summary>View the final-results graphic</summary>

<img src="assets/raceday_images/race_result.avif" alt="JetRacer university final results listing ZENITH RACING as first runner-up" width="650">

</details>

## Race footage

### Qualifying session

<img src="assets/raceday_images/video1.gif" alt="Archived footage from the qualifying session" width="850">

### First round of the final

<img src="assets/raceday_images/video2.gif" alt="Archived final-round footage showing the car leaving the track" width="850">

The original race report identifies this clip as an off-track incident in the first round of the final. It documents a failure case alongside the successful competition result.

## Project scope and code provenance

This repository was assembled after the competition to preserve the event materials and provide an entry point to the JetRacer workflow.

| Material | Source and scope |
| --- | --- |
| Race photographs, GIFs, and result graphic | Competition records archived in this repository |
| JetRacer and JetBot STL files | Archived hardware resources; individual design authorship is not documented here |
| Competition rules and master-class slides | Organizer materials retained as reference documents |
| Control classes, training notebooks, and TensorRT examples | [NVIDIA JetRacer](https://github.com/NVIDIA-AI-IOT/jetracer), included as a Git submodule at [`b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543`](https://github.com/NVIDIA-AI-IOT/jetracer/tree/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543) |

The public snapshot contains no separate team-specific control implementation, training images, trained model weights, or experiment logs. The technical details below describe the included upstream examples; the exact configuration used during the race is not recorded.

## Technical workflow

The included examples use supervised image-coordinate regression. An operator labels a target point on the intended driving path; a neural network predicts that point from a camera image, and its horizontal coordinate determines steering.

| Stage | Implementation in the included code |
| --- | --- |
| Capture and labeling | JetCam CSI camera; click a target point in a Jupyter image widget |
| Dataset | JPEG filenames encode pixel coordinates as `x_y_uuid.jpg`, grouped under the `apex` category |
| Preprocessing | 224 × 224 images, tensor conversion, ImageNet mean and standard deviation |
| Training augmentation | Color jitter and random horizontal flips with the horizontal label sign reversed |
| Model | Pretrained ResNet-18 with its final layer replaced by a two-coordinate regression head |
| Training | Adam optimizer, batch size 8, mean squared error over the target coordinates |
| Deployment | `torch2trt` conversion with FP16 enabled, then inference through a TensorRT module |
| Actuation | Predicted horizontal coordinate mapped to steering; throttle set separately |

The dataset helper normalizes a pixel target using:

```text
x_normalized = 2 × (x_pixel / image_width  − 0.5)
y_normalized = 2 × (y_pixel / image_height − 0.5)
```

In [`road_following.ipynb`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/road_following.ipynb), the inference loop applies:

```python
car.steering = x * STEERING_GAIN + STEERING_BIAS
```

The [`NvidiaRacecar`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/jetracer/nvidia_racecar.py) class then applies the hardware steering gain and offset before writing to the servo. These are two separate calibration stages. The road-following example uses the predicted `x` for steering and a fixed throttle command; it does not implement an explicit overtaking planner or opponent detector.

Sources: [`interactive_regression.ipynb`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/interactive_regression.ipynb), [`xy_dataset.py`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/xy_dataset.py), [`utils.py`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/utils.py), and [`road_following.ipynb`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/road_following.ipynb).

## Competition observations

The [archived race rules](<docs/competition_rules/Formula Edge Macau 2024 University Schedule and Game Rules.pdf>) cover multi-car racing, overtaking, contact, and off-track outcomes. Pages 18–25 describe overtaking responsibility and collision outcomes: the following car must avoid the leading car, and penalties depend on relative position and whether the cars can continue.

These conditions introduce practical challenges for camera-based road following:

- Target-point labels determine the path the model learns. Labeling should account for track boundaries, turns, and recovery from lateral or heading errors.
- Other cars can obscure the track and change the camera input. Successful solo driving alone does not establish reliable behavior in traffic.
- Steering calibration and throttle affect whether a predicted path can be followed. They need to be checked on the physical vehicle.
- An off-track incident is useful evidence for further data collection and testing. The archived footage does not establish its technical cause.

These are qualitative considerations from the workflow and competition conditions. This archive provides no measured lap times, inference benchmarks, held-out evaluation results, or controlled comparisons.

## Getting started

### 1. Clone with the upstream code

```bash
git clone --recurse-submodules https://github.com/tingyangxie/FormulaEdge-Macau-2024.git
cd FormulaEdge-Macau-2024
```

For an existing clone:

```bash
git submodule update --init --recursive
```

The `jetracer/` directory is a submodule. A normal source ZIP does not include its contents; use a recursive clone to obtain the pinned code.

### 2. Prepare the Jetson environment

Follow the included [software setup](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/docs/software_setup.md) and the instructions for your specific car kit. The upstream documentation describes a historical Jetson Nano / JetPack 4.5.1 environment. The examples depend on Jetson-compatible PyTorch and torchvision, CUDA/TensorRT, JetCam, `torch2trt`, ServoKit, and Jupyter widgets, including the clickable-image widget used for labeling.

These hardware-dependent notebooks are not ready to run on a standard desktop Python installation. The original competition environment is not preserved as a lockfile or system image, and the notebooks have not been revalidated here on hardware.

### 3. Follow the notebook sequence

Open `jetracer/notebooks/` in Jupyter on the Jetson, so that the local helper modules are available.

| Order | Notebook | Purpose |
| --- | --- | --- |
| 1 | [`basic_motion.ipynb`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/basic_motion.ipynb) | Check steering direction, neutral position, and throttle response |
| 2 | [`interactive_regression.ipynb`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/interactive_regression.ipynb) | Collect and label your own track images, train, and save `road_following_model.pth` |
| 3 | [`road_following.ipynb`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/road_following.ipynb) | Load that checkpoint, convert it to TensorRT, save `road_following_model_trt.pth`, and run road following |

An alternative split workflow uses [`optimize_model.ipynb`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/optimize_model.ipynb) followed by [`road_following_live.ipynb`](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/notebooks/road_following_live.ipynb). Those files default to `model_08-06.pth` and `model_08-06_trt.pth`; change their paths to match the checkpoints you actually save. These model files are not included in this repository.

Before executing motion cells, lift the wheels off the ground to check control direction and neutral throttle. The `road_following.ipynb` loop has no automatic stop handler: interrupting the cell does not itself issue a zero-throttle command. Set `car.throttle = 0.0` when stopping, and keep a physical means of disconnecting motor power available.

## Hardware and 3D printing

The repository contains STL files and matching previews for several JetRacer components. The filenames identify the archived parts; they do not establish which combination was fitted to the race car.

| JetRacer resource | STL | Preview |
| --- | --- | --- |
| Camera mount, labeled 15° | [Download](docs/JetRacer_hardware_3Dprint/jr_cam_mount_15deg-Body.stl) | [View](assets/jetracer_hardware_diycomponent/jr_cam_mount_15deg-Body.png) |
| Baseboard, labeled v3 | [Download](docs/JetRacer_hardware_3Dprint/jr_baseboard_for_v3-Body.stl) | [View](assets/jetracer_hardware_diycomponent/jr_baseboard_for_v3-Body.png) |
| Antenna mount | [Download](docs/JetRacer_hardware_3Dprint/jr_antenna_mount-Body.stl) | [View](assets/jetracer_hardware_diycomponent/jr_antenna_mount-Body.png) |
| TT02 adapter board | [Download](docs/JetRacer_hardware_3Dprint/jr_adaptor_brd_tt02-Body.stl) | [View](assets/jetracer_hardware_diycomponent/jr_adaptor_brd_tt02-Body.png) |
| M20 adapter board | [Download](docs/JetRacer_hardware_3Dprint/jr_adaptor_brd_M20-Body.stl) | [View](assets/jetracer_hardware_diycomponent/jr_adaptor_brd_M20-Body.png) |
| DGX adapter board | [Download](docs/JetRacer_hardware_3Dprint/jr_adaptor_brd_DGX-Body.stl) | [View](assets/jetracer_hardware_diycomponent/jr_adaptor_brd_DGX-Body.png) |

Additional [JetBot STL files](docs/JetBot_hardware_3Dprint/) are retained as related hardware resources. Printing material, tolerances, infill, and fit checks are not documented. For the upstream reference builds, see the [Tamiya TT02 hardware guide](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/docs/tamiya/hardware_setup.md) and [Latrax hardware guide](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/docs/latrax/hardware_setup.md).

## Repository contents

```text
FormulaEdge-Macau-2024/
├── README.md
├── LICENSE
├── .gitmodules                         # NVIDIA JetRacer submodule source
├── assets/
│   ├── raceday_images/                 # Photos, race GIFs, and final results
│   └── jetracer_hardware_diycomponent/  # Printable-part previews
├── docs/
│   ├── competition_rules/              # Rules, schedule, and master-class PDFs
│   ├── JetRacer_hardware_3Dprint/       # JetRacer STL files
│   └── JetBot_hardware_3Dprint/         # Related JetBot STL files
└── jetracer/                           # Pinned NVIDIA upstream submodule
    ├── jetracer/                       # Racecar control classes
    ├── notebooks/                      # Labeling, training, and deployment
    └── docs/                           # Upstream setup and hardware guides
```

## Reference documents

- [Competition schedule and on-track rules (English)](<docs/competition_rules/Formula Edge Macau 2024 University Schedule and Game Rules.pdf>)
- [Competition format and rules (Traditional Chinese)](<docs/competition_rules/校際人工智能 Edge 級方程式比賽 - 大灣區 - 澳門站2024 - 大學 - 比賽規則.pdf>)
- [JetRacer university master-class slides](<docs/competition_rules/澳門JetRacer大學master class 1.pdf>)
- [NVIDIA JetRacer upstream repository](https://github.com/NVIDIA-AI-IOT/jetracer)

<details>
<summary>More competition photographs</summary>

<table>
  <tr>
    <td><img src="assets/raceday_images/IMG_5815.jpg" alt="Competition-day photograph 5815" width="420"></td>
    <td><img src="assets/raceday_images/IMG_5816.jpg" alt="Competition-day photograph 5816" width="420"></td>
  </tr>
  <tr>
    <td><img src="assets/raceday_images/IMG_5803.jpg" alt="Competition-day photograph 5803" width="420"></td>
    <td><img src="assets/raceday_images/IMG_5832.jpg" alt="Competition-day photograph 5832" width="420"></td>
  </tr>
</table>

[Team identification and car preparation](assets/raceday_images/IMG_5801.jpg) · [Additional photograph](assets/raceday_images/IMG_5829.jpg)

</details>

## License and acknowledgments

The repository's own material is covered by the [MIT License](LICENSE). NVIDIA JetRacer retains its [upstream MIT license and copyright notice](https://github.com/NVIDIA-AI-IOT/jetracer/blob/b194ffd3477ce4e7b8bcf5a9cd83b3dc79371543/LICENSE.md). Competition documents and third-party resources retain their respective notices; their inclusion does not establish that they are MIT-licensed.

Acknowledgments to the Formula Edge organizers and STEM PLUS for the competition and teaching materials, and to NVIDIA's JetRacer contributors for the hardware-control and learning examples.
