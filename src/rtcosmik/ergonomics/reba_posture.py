"""Posture-only REBA from RT-COSMIK joint angles and markers, vectorized over frames.

Implements the Rapid Entire Body Assessment (Hignett & McAtamney, Applied
Ergonomics 31(2), 2000) for every frame of a trial. Tables A, B and C are the
published tables. The inputs are what needs care, because a video pipeline does
not observe everything REBA asks for, and the model's joint angles are not
anatomical angles:

* **Posture only.** Load/force (step 5), coupling (step 11) and activity (step 14)
  are zero: nothing in the data measures them. The result is a posture REBA, and
  should be called that.
* **Wrists score 1.** The parity marker set locks both wrist DoF.
* **Legs assume bilateral support** (there is no foot-contact estimate). Knee
  flexion adds +1 for 30-60 deg and +2 beyond 60; the more flexed knee counts.
* **Trunk flexion is inclination from the vertical**, of the pelvis-centre-to-C7
  axis, signed forward in the subject's own sagittal plane. REBA scores how far
  the trunk leans, not the pelvis-trunk joint angle; the two differ whenever the
  pelvis tilts, which is exactly what lifting and squatting do.
* **Angles are relative to a neutral posture.** The model's joint zeros are not
  anatomical zeros: on the mocap reference, cervical flexion has a median of +22
  deg during overhead work, where people look up, and elbow flexion never falls
  below ~37 deg. Absolute REBA thresholds only make sense after the participant's
  neutral posture is subtracted. Every COMFI trial starts with the participant
  standing in the calibration pose, so the neutral is taken there, per trial.
* **Thresholds REBA leaves unspecified** ("twisted or side-bent", "abducted",
  "shoulder raised") keep the values of the toolbox's earlier implementation:
  15 deg for the neck, 5 deg for the trunk, 15 deg of shoulder abduction, and a
  shoulder-to-ASIS distance 25 mm above neutral.
"""
import numpy as np

TABLE_A = np.array([  # [neck, trunk, legs]
    [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
    [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
])
TABLE_B = np.array([  # [upper arm, lower arm, wrist]
    [[1, 2, 2], [1, 2, 3]], [[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 5]],
    [[4, 5, 5], [5, 6, 7]], [[6, 7, 8], [7, 8, 8]], [[7, 8, 8], [8, 9, 9]],
])
TABLE_C = np.array([  # [score A, score B]
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7], [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8], [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9], [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11], [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12], [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12], [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
])

#: REBA action levels: 1 negligible, 2-3 low, 4-7 medium, 8-10 high, 11+ very high.
RISK_EDGES = (1, 3, 7, 10)
RISK_NAMES = ("negligible", "low", "medium", "high", "very high")

NECK_TWIST_DEG, TRUNK_TWIST_DEG = 15.0, 5.0
ABDUCTION_DEG, SHOULDER_RAISE_M = 15.0, 0.025

#: Joint angles REBA reads, by our CSV column names.
ANGLES = {
    "neck_flex": "Cervical_Flexion_Extension[rad]",
    "neck_side": "Cervical_Lateral_Bending[rad]",
    "neck_rot": "Cervical_Internal_External_Rotation[rad]",
    "trunk_side": "Lumbar_Lateral_Bending[rad]",
    "r_knee": "Right_Knee_Flexion_Extension[rad]",
    "l_knee": "Left_Knee_Flexion_Extension[rad]",
    "r_sh_flex": "Right_Shoulder_Flexion_Extension[rad]",
    "l_sh_flex": "Left_Shoulder_Flexion_Extension[rad]",
    "r_sh_abd": "Right_Shoulder_Abduction_Adduction[rad]",
    "l_sh_abd": "Left_Shoulder_Abduction_Adduction[rad]",
    "r_elbow": "Right_Elbow_Flexion_Extension[rad]",
    "l_elbow": "Left_Elbow_Flexion_Extension[rad]",
}
PELVIS = ("RASI", "LASI", "RPSI", "LPSI")


def _marker(markers, name):
    return np.column_stack([markers[f"{name}_{a}"] for a in "xyz"]).astype(float)


def raw_inputs(joints, markers):
    """Per-frame REBA inputs before neutral correction: degrees and metres.

    ``joints`` and ``markers`` are mappings from CSV column name to array, of
    equal length (e.g. pandas DataFrames already trimmed to the same frames).
    """
    out = {key: np.degrees(np.asarray(joints[col], dtype=float)) for key, col in ANGLES.items()}

    asis = (_marker(markers, "RASI") + _marker(markers, "LASI")) / 2
    psis = (_marker(markers, "RPSI") + _marker(markers, "LPSI")) / 2
    pelvis = (asis + psis) / 2
    axis = _marker(markers, "C7") - pelvis
    forward = asis - psis
    forward[:, 2] = 0.0                                   # world z is up
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    out["trunk_incl"] = np.degrees(np.arctan2((axis * forward).sum(1), axis[:, 2]))

    out["r_raise"] = np.linalg.norm(_marker(markers, "RSHO") - _marker(markers, "RASI"), axis=1)
    out["l_raise"] = np.linalg.norm(_marker(markers, "LSHO") - _marker(markers, "LASI"), axis=1)
    return out


#: Frames of a trial's calibration pose: its first 0.5 s at 40 Hz. On the mocap
#: reference, no trial starts moving before 0.75 s (median 1.7 s), and over this
#: window the neutral agrees with the one from COMFI's separate Static trial to
#: -0.4 (SD 1.3) deg of trunk inclination and 1.0 (SD 5.6) deg of elbow flexion.
NEUTRAL_FRAMES = slice(0, 20)


def neutral_from(joints, markers):
    """A neutral posture: median of every input over the frames given, typically
    a trial's first ``NEUTRAL_FRAMES``."""
    return {key: float(np.nanmedian(values)) for key, values in raw_inputs(joints, markers).items()}


def scores(joints, markers, neutral):
    """Per-frame REBA components, score and risk level, relative to ``neutral``."""
    x = {key: values - neutral[key] for key, values in raw_inputs(joints, markers).items()}

    neck = np.where((x["neck_flex"] > 20) | (x["neck_flex"] < 0), 2, 1)
    neck = neck + ((np.abs(x["neck_side"]) > NECK_TWIST_DEG)
                   | (np.abs(x["neck_rot"]) > NECK_TWIST_DEG))

    t = x["trunk_incl"]
    trunk = np.select([np.abs(t) <= 5, (t > -20) & (t < 20), ((t >= 20) & (t < 60)) | (t <= -20)],
                      [1, 2, 3], default=4)
    trunk = trunk + (np.abs(x["trunk_side"]) > TRUNK_TWIST_DEG)

    knee = np.maximum(x["r_knee"], x["l_knee"])
    legs = 1 + np.select([knee > 60, knee > 30], [2, 1], default=0)

    def upper(flex, abd, raise_):
        base = np.select([(flex >= -20) & (flex <= 20), (flex < -20) | (flex <= 45), flex <= 90],
                         [1, 2, 3], default=4)
        return np.clip(base + (abd > ABDUCTION_DEG) + (raise_ > SHOULDER_RAISE_M), 1, 6)
    upper_arm = np.maximum(upper(x["r_sh_flex"], x["r_sh_abd"], x["r_raise"]),
                           upper(x["l_sh_flex"], x["l_sh_abd"], x["l_raise"]))

    def lower(elbow):
        return np.where((elbow >= 60) & (elbow <= 100), 1, 2)
    lower_arm = np.maximum(lower(x["r_elbow"]), lower(x["l_elbow"]))
    wrist = np.ones_like(neck)

    ok = np.all([np.isfinite(v) for v in x.values()], axis=0)
    score_a = TABLE_A[neck - 1, trunk - 1, legs - 1]
    score_b = TABLE_B[upper_arm - 1, lower_arm - 1, wrist - 1]
    reba = TABLE_C[score_a - 1, score_b - 1].astype(float)
    reba[~ok] = np.nan
    risk = np.digitize(reba, RISK_EDGES, right=True).astype(float)
    risk[~ok] = np.nan
    return {"neck": neck, "trunk": trunk, "legs": legs, "upper_arm": upper_arm,
            "lower_arm": lower_arm, "wrist": wrist, "score_a": score_a,
            "score_b": score_b, "reba": reba, "risk": risk, "valid": ok}
