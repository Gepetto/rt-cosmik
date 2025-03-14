import numpy as np
from numpy import linalg as LA
from scipy import signal
import cv2

def trace(m):
    return float(np.trace(m))

def rotmat_from_values(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    return np.array([[x0, x1, x2], 
                     [y0, y1, y2], 
                     [z0, z1, z2]], dtype=np.float64)

def rotmat_from_row_vecs(x, y, z):
    return np.array([[x[0], x[1], x[2]], 
                     [y[0], y[1], y[2]], 
                     [z[0], z[1], z[2]]], dtype=np.float64)

def identity_3D():
    return np.eye(3, dtype=np.float64)

def norm(vector):
    return LA.norm(vector)

def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=np.float64)

def row_vector_3D(a, b, c):
    return np.array([float(a), float(b), float(c)], dtype=np.float64)

def col_vector_3D_from_tab(x):
    return np.array([[float(x[0])], [float(x[1])], [float(x[2])]])

def rotmat(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    return np.array([[x0, x1, x2], 
                     [y0, y1, y2], 
                     [z0, z1, z2]]).astype(float)

def RMSE(est, ref):
    sq_err_sum=0
    for i in range(len(est)):
        sq_err_sum += pow(est[i] - ref[i], 2)
    
    rmse = np.sqrt(sq_err_sum/len(est))
    return rmse

def vec_to_skewmat(x):
    return np.array([[0.0, -x[2], x[1]], 
                     [x[2], 0.0, -x[0]], 
                     [-x[1], x[0], 0.0]], dtype=object).astype(float)
                     
def skewmat_to_vec(skew):
    return col_vector_3D(-skew[1][2], skew[0][2], -skew[0][1])

def rot_to_cayley(rot):
    """Convert rotation matrix to Cayley representation."""
    cayley_skew = (identity_3D() - rot) @ np.linalg.inv(identity_3D() + rot)
    return skewmat_to_vec(cayley_skew)

def cayley_to_rot(cayley):
    return (identity_3D() - vec_to_skewmat(cayley))*np.linalg.inv((identity_3D() + vec_to_skewmat(cayley)))

def make_homogeneous_rep_matrix(R, t):
    if t.shape not in [(3,), (3, 1)]:
        raise ValueError("Translation vector must be of shape (3,) or (3,1)")
    P = np.eye(4, dtype=np.float64)
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    return P

def butterworth_filter(data, cutoff_frequency, order=5, sampling_frequency=60):
    nyquist = 0.5 * sampling_frequency
    if not 0 < cutoff_frequency < nyquist:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")
    b, a = signal.butter(order, cutoff_frequency / nyquist, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


def low_pass_filter_data(data,nbutter=5):
    '''This function filters and elaborates data used in the identification process. 
    It is based on a return of experience  of Prof Maxime Gautier (LS2N, Nantes, France)'''
    
    b, a = signal.butter(nbutter, 0.01*5 / 2, "low")
   
    #data = signal.medfilt(data, 3)
    data= signal.filtfilt(
            b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1) )
    
    
    # suppress end segments of samples due to the border effect
    # nbord = 5 * nbutter
    # data = np.delete(data, np.s_[0:nbord], axis=0)
    # data = np.delete(data, np.s_[(data.shape[0] - nbord): data.shape[0]], axis=0)
     
    return data

def concat_frames(frames):
    if len(frames) == 2:
        return cv2.hconcat(frames)
    elif len(frames) == 4:
        # Horizontally concatenate pairs
        h_top = cv2.hconcat(frames[:2])
        h_bottom = cv2.hconcat(frames[2:])
        # Vertically concatenate the two rows
        return cv2.vconcat([h_top, h_bottom])
    else:
        raise ValueError("Only 2 or 4 frames are supported for concatenation.")



def reproject(results, frame_size, axis="horizontal"):
    """
    Reprojects detected keypoints and bounding boxes to their original frames
    after concatenation along a specified axis.

    Parameters:
    - results: tuple (keypoints, bboxes, _), output of the pose estimator.
    - frame_size: int, width (if horizontal) or height (if vertical) of one original frame.
    - axis: str, either "horizontal" (x-axis) or "vertical" (y-axis).

    Returns:
    - first_result: (keypoints, bbox, _), corresponding to the first original frame.
    - second_result: (keypoints, bbox, _), corresponding to the second original frame.
    """
    keypoints, bboxes, _ = results

    if len(keypoints) < 2 :
        return results, results  # Not enough skeletons detected
    else :
        coord_idx = 0 if axis == "horizontal" else 1  # 0 for x, 1 for y

        # Compute mean coordinates (x or y)
        mean_values = [kp[:, coord_idx].mean() for kp in keypoints]
        first_idx = np.argmin(mean_values)   # Left (if horizontal) or Top (if vertical)
        second_idx = np.argmax(mean_values)  # Right (if horizontal) or Bottom (if vertical)

        first_skeleton = keypoints[[first_idx]]
        first_bboxes = bboxes[first_idx]

        second_skeleton = keypoints[[second_idx]]
        second_bboxes = bboxes[second_idx]

        # Shift second skeleton back to its original frame coordinates
        second_skeleton[..., coord_idx] -= frame_size

        first_result = (first_skeleton, first_bboxes, _)
        second_result = (second_skeleton, second_bboxes, _)

        return [first_result, second_result]


def reproject_four_frames(results, frame_width, frame_height):
    """
    Reprojects detected keypoints and bounding boxes to their original frames
    after two-step concatenation: horizontal + vertical.

    Parameters:
    - results: tuple (keypoints, bboxes, _), output of the pose estimator.
    - frame_width: int, width of a single original frame.
    - frame_height: int, height of a single original frame.

    Returns:
    - 4 results
    """
    keypoints, bboxes, _ = results

    if len(keypoints) < 4 :
        return results, results, results, results  # Not enough skeletons detected
    else :
        # Step 1: Separate into top and bottom stacked frames
        mean_y_values = [kp[:, 1].mean() for kp in keypoints]
        top_indices = np.argsort(mean_y_values)[:2]    # Two skeletons with smallest y (top row)
        bottom_indices = np.argsort(mean_y_values)[2:] # Two skeletons with largest y (bottom row)

        top_skeletons = keypoints[top_indices]
        top_bboxes = bboxes[top_indices]
        
        bottom_skeletons = keypoints[bottom_indices]
        bottom_bboxes = bboxes[bottom_indices]

        # Adjust bottom skeletons back to their original y-coordinates
        bottom_skeletons[..., 1] -= frame_height

        # Step 2: Separate left and right within top and bottom stacked frames
        def split_horizontally(skeletons, bboxes):
            """Splits horizontally stacked frames"""
            mean_x_values = [kp[:, 0].mean() for kp in skeletons]
            left_idx = np.argmin(mean_x_values)   # Left frame
            right_idx = np.argmax(mean_x_values)  # Right frame
            
            left_skeleton = skeletons[[left_idx]]
            left_bboxes = bboxes[left_idx]

            right_skeleton = skeletons[[right_idx]]
            right_bboxes = bboxes[right_idx]

            # Adjust right skeletons back to their original x-coordinates
            right_skeleton[..., 0] -= frame_width

            return (left_skeleton, left_bboxes, _), (right_skeleton, right_bboxes, _)

        # Split top stacked frame into left and right
        top_left_result, top_right_result = split_horizontally(top_skeletons, top_bboxes)
        
        # Split bottom stacked frame into left and right
        bottom_left_result, bottom_right_result = split_horizontally(bottom_skeletons, bottom_bboxes)

        # Return dictionary containing results for each original frame
        return [top_left_result,top_right_result,bottom_left_result,bottom_right_result]


