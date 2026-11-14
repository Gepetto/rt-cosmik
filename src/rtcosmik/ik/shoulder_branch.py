"""The shoulder's two angle triples for one posture: ``(z, x, y)`` and ``(z + 180, 180 - x, y + 180)``."""
import numpy as np

SIDES = ("left", "right")
AXES = "ZXY"
MARGIN = np.radians(3.0)     # how close to a limit counts as pinned
HOLD_FRAMES = 20             # frames a moved shoulder is left alone


def wrap(angle):
    """Angles brought into [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def other_branch(triple):
    """The other ``(z, x, y)`` of the same rotation; rows are the three angles."""
    z, x, y = triple
    return np.stack([wrap(z + np.pi), wrap(np.pi - x), wrap(y + np.pi)])


def nearest_branch(triple, reference):
    """``triple`` in whichever of its two branches is closer to ``reference``; NaN kept."""
    other = other_branch(triple)
    distance = lambda t: np.sum(wrap(t - reference) ** 2, axis=0)
    return np.where(distance(other) < distance(triple), other, triple)


def columns(model, side):
    """Position and velocity indices of one shoulder's three joints, in Z, X, Y order."""
    joints = [model.joints[model.getJointId(f"{side}_shoulder_{axis}")] for axis in AXES]
    return [j.idx_q for j in joints], [j.idx_v for j in joints]


class BranchReseed:
    """Move a horizon whose shoulder is pinned on a limit to the branch that is free."""

    def __init__(self, model):
        self.nq = model.nq
        self.shoulders = []
        for side in SIDES:
            cq, cv = columns(model, side)
            self.shoulders.append((side, cq, cv, np.array(model.lowerPositionLimit[cq]),
                                   np.array(model.upperPositionLimit[cq])))
        self.frames_since_move = {side: HOLD_FRAMES for side in SIDES}

    def __call__(self, X, U):
        """Rewrite states ``X`` (nx, N) and controls ``U`` (nv, N) in place; return the sides moved."""
        moved = []
        for side, cq, cv, low, high in self.shoulders:
            self.frames_since_move[side] += 1
            current = np.array(X[cq, :])
            newest = current[:, -1]
            pinned = np.any((newest < low + MARGIN) | (newest > high - MARGIN))
            if not pinned or self.frames_since_move[side] <= HOLD_FRAMES:
                continue
            other = other_branch(current)
            if not np.all((other > low[:, None] + MARGIN) & (other < high[:, None] - MARGIN)):
                continue
            X[cq, :] = other
            X[self.nq + cv[1], :] *= -1          # abduction runs the other way
            U[cv[1], :] *= -1
            self.frames_since_move[side] = 0
            moved.append(side)
        return moved
