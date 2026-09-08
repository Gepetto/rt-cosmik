import numpy as np
import pinocchio as pin
import meshcat
import hppfcl
import os
# Meshcat utils
from meshcat.geometry import Cylinder, MeshLambertMaterial
from meshcat.transformations import rotation_matrix, translation_matrix, concatenate_matrices



def meshcat_material(r, g, b, a):
    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + int(b * 255)
    material.opacity = a
    return material


def meshcat_transform(x, y, z, q, u, a, t):
    return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))


# Gepetto/meshcat abstraction

def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Box([sizex, sizey, sizez]),
                                    meshcat_material(*rgba))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addBox(name, sizex, sizey, sizez, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def addViewerSphere(viz, name, size, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Sphere(size),
                                    meshcat_material(*rgba))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addSphere(name, size, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def applyViewerConfiguration(viz, name, xyzquat):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_transform(meshcat_transform(*xyzquat))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.applyConfiguration(name, xyzquat)
        viz.viewer.gui.refresh()
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)



def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with
    vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix

# def place(viz, name, M):
#     """This function places in the gui a coordinate system at the location provided in M.
#     Input: viz (viz) a robot visualiser (such as gepetto-viewer)
#            name (str) the name of the object coordinate system
#            M (se3) homogenous transformation matrix
#     Output: (void) places the object at the desired location
#     """
#     viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(M))
# def display_force(viz, phi, M_se3):
#     """Displays the force phi (expressed in the M_se3 frame) in the M_se3 frame and the
#     visualiser viz given
#     Input: viz (viz) a pinocchio visualiser
#             phi (Force Tpl) a pinocchio force 6D vector
#             M_se3 (SE3) the se3 object relating the transformation matrix in which we
#             want the force to be displayed
#     Output: (void) Displays the force in the gui
#     """
#     M_se3_temp = M_se3
#     color = [1, 1, 0, 1]
#     radius = 0.01
#     phi = phi.se3Action(M_se3)
#     force = [phi.linear[0], phi.linear[1], phi.linear[2]]
#     length = np.linalg.norm(force)*1e-3
#     Rot = rotation_matrix_from_vectors([1, 0, 0], phi.linear)   # addArrow in pinocchio is always along the x_axis so we have to project the x_axis on the direction of the force vector for display purposes # noqa
#     M_se3_temp.rotation = Rot
#     viz.viewer.gui.addArrow("world/arrow", radius, length, color)
#     place(viz, "world/arrow", M_se3_temp)
def compute_cop(phi: pin.Force, contact_frame: pin.SE3):
    """
    Compute Center of Pressure (CoP) in world coordinates from a contact wrench.
    # Phi is expressed in the foot frame with Y pointing upwards, X front and Z lateral
    """
    f = phi.linear
    m = phi.angular
    
     
    f_global = contact_frame.act(phi)
    f_lin = f_global.linear   # casadi.SX or MX with shape (3,)
    f_ang = f_global.angular  # same shape
     
    cop_x = -f_ang [1] / f_lin[2]
    cop_y = f_ang[0] /f_lin[2]
    
    cop_world = np.array([cop_x, cop_y, contact_frame.translation[2]])
    # # Avoid division by zero
    # fz = f[1]
    # if abs(fz) < 1e-6:
    #     return contact_frame.translation

    # # CoP expressed in the global frame (XY components only)
    # cop_local_x = m[2] / fz
    # cop_local_y = -m[0] / fz
    # cop_local = np.array([cop_local_x, cop_local_y, 0.0])
    

    # print(cop_local)
    # # Transform CoP to world frame
    # cop_world = contact_frame.act(cop_local)
    return cop_world

def display_force(viz, phi, M_se3):
    """
    Display a 6D force as an arrow, starting at the center of pressure, oriented along the force.
    """
    color = [1, 1, 0, 1]
    radius = 0.01

    # Compute CoP in world frame
    cop_world = compute_cop(phi, M_se3)

    # Compute arrow direction from force
    force_vec = phi.linear
    length = np.linalg.norm(force_vec) * 1e-3
    if length < 1e-6:
        return  # Avoid rendering zero-length force

    # Normalize and compute orientation
    direction = force_vec / np.linalg.norm(force_vec)
    Rot = rotation_matrix_from_vectors([0, 0, 1], direction)

    # Build transformation from CoP
    M_arrow = pin.SE3(Rot, cop_world)

    # Create and display arrow
    arrow = meshcat.geometry.Cylinder(height=length, radius=radius)
    mat = meshcat.geometry.MeshLambertMaterial(color=0xFFFF00)
    viz.viewer["world/force_arrow"].set_object(arrow, mat)
   
   # First translate the cylinder so its base will start at the origin
    T_shift = translation_along_z(length / 2)

    # Then apply the rotation and translation to position it at the CoP
     
    R_align = rotation_matrix_from_vectors([0, 0, 1], direction)
    M_arrow.rotation = R_align

    # Final transform for Meshcat
    T_final = pose_to_matrix(M_arrow) @ T_shift
    viz.viewer["world/force_arrow"].set_transform(T_final)
   
   

def pose_to_matrix(se3_obj: pin.SE3) -> np.ndarray:
    """
    Convert a Pinocchio SE3 object to a 4x4 homogeneous transformation matrix.
    """
    mat = np.eye(4)
    mat[:3, :3] = se3_obj.rotation
    mat[:3, 3] = se3_obj.translation
    return mat

def translation_along_z(dz: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = dz
    return T

def rotation_matrix_from_vectors(a, b):
    """Returns rotation matrix that aligns vector a to b."""
    a = np.array(a) / np.linalg.norm(a)
    b = np.array(b) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, 1):
        return np.eye(3)
    elif np.isclose(c, -1):
        # 180 degree rotation: pick orthogonal axis
        orth = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        v = np.cross(a, orth)
        v = v / np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        return np.eye(3) + 2 * kmat @ kmat
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))


            
def display_com(model, data, viz, q):
    # compute CoMs
    pin.centerOfMass(model, data, q)

    # all joint names
    all_joint_names = [model.names[j] for j in range(model.njoints)]

    for i, jn in enumerate(all_joint_names):
        if not model.existJointName(jn):
            continue

        joint_id = model.getJointId(jn)

        # --- mass and radius ---
        if i == 0:  # total CoM
            mass = sum([Y.mass for Y in model.inertias[1:]])  # exclude universe
            radius = 0.05
            color = [0.5, 0.5, 0.0, 1]
        else:  # segment CoM
            mass = model.inertias[joint_id].mass
            radius = 0.01 + 0.002 * mass   # scale sphere radius with mass
            color = [0.2, 0.6, 0.9, 1]

        # add sphere if not already present
        addViewerSphere(viz, f'com_{i}', radius, color)

        # get CoM placement
        placement = data.oMi[joint_id]
        com_pos = placement.act(data.com[i])

        applyViewerConfiguration(
            viz,
            f'com_{i}',
            np.hstack((com_pos, np.array([0, 0, 0, 1])))
        )

        # total CoM floor projection
        if i == 0:
            addViewerSphere(viz, f'com_floor_{i}', radius, [1, 0, 0, 0.6])
            com_floor = com_pos.copy()
            com_floor[2] = -1
            applyViewerConfiguration(
                viz,
                f'com_floor_{i}',
                np.hstack((com_floor, np.array([0, 0, 0, 1])))
            )        
        

def display_model_frames(model, visual_model, frame2display, param) :
    # This function displays 3D frames at selected frame locations
    # frame2display: list containing frames names to be displayed
    factor=0.5
    scaling=[factor*0.0025,factor*0.0025,factor*0.0025]
    meshloader=hppfcl.MeshLoader()
    
    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Go one folder back
    parent_directory = os.path.dirname(script_directory)
    
     
       # modify color for animation
    # for i in range(len(visual_model.geometryObjects)):
    #     visual_model.geometryObjects[i].meshColor = np.array([0.05, 0.8, 0.2, 0.5])       
     
    # create visual XYZ frames at frames of interest 
    urdf_meshes_path = os.path.join(parent_directory, "model")
    X = pin.utils.rotate("y", np.pi / 2)
    Y = pin.utils.rotate("x", -np.pi / 2)
    Z = np.eye(3) 
    
    # print("All frame names:")
    # for i, frame in enumerate(model.frames):
    #     print(f"Frame {i}: {frame.name}")
    
 
    for i in range(len(frame2display)):#len(param.active_joints)):
        
       
        #print(frame2display[i])
        # add a XYZ frame at the the desired frame
         
        FIDX = model.getFrameId(frame2display[i]) 
        
        #print(FIDX)
        # for i, frame in enumerate(model.frames):
        #     print(f"[{i}] Frame name: {frame.name}, parent joint: {frame.parent}, type: {frame.type}") 
 
        #print(model.frames[FIDX].parent)
        JIDX = model.frames[FIDX].parent

        X = pin.utils.rotate("y", np.pi / 2)
        Y = pin.utils.rotate("x", -np.pi / 2)
        Z = np.eye(3)

        position = model.frames[FIDX].placement.translation# np.array([0, 0, 0])
        
        
        visual_model.addGeometryObject(pin.GeometryObject(f'frames_axis_x_{i}', FIDX, JIDX, meshloader.load("model/human_urdf/meshes/X_arrow.stl"),pin.SE3(X, position),"model/human_urdf/meshes/X_arrow.stl",np.array(scaling) ))
        visual_model.geometryObjects[-1].meshColor = np.array([1, 0, 0, 1.0])

        visual_model.addGeometryObject(pin.GeometryObject(f'frames_axis_y_{i}', FIDX, JIDX, meshloader.load("model/human_urdf/meshes/Y_arrow.stl"),pin.SE3(Y, position),"model/human_urdf/meshes/Y_arrow.stl",np.array(scaling) ))
        visual_model.geometryObjects[-1].meshColor = np.array([0, 1, 0, 1.0])

        visual_model.addGeometryObject(pin.GeometryObject(f'frames_axis_z_{i}', FIDX, JIDX, meshloader.load("model/human_urdf/meshes/Z_arrow.stl"),pin.SE3(Z, position),"model/human_urdf/meshes/Y_arrow.stl",np.array(scaling) ))
        visual_model.geometryObjects[-1].meshColor = np.array([0, 0, 1, 1.0])     

        
        # #Frames attaches to the model
        # visual_model.addGeometryObject(pin.GeometryObject(f'frames_axis_x_{i}', JIDX, pin.SE3(X, position), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/X_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/X_arrow.stl",np.array([0.00025,0.00025,0.00025]) ))
        # visual_model.geometryObjects[-1].meshColor = np.array([1, 0, 0, 1.0])

        # visual_model.addGeometryObject(pin.GeometryObject(f'frames_axis_y_{i}', JIDX, pin.SE3(Y, position), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl",np.array([0.00025,0.00025,0.00025]) ))
        # visual_model.geometryObjects[-1].meshColor = np.array([0, 1, 0, 1.0])

        # visual_model.addGeometryObject(pin.GeometryObject(f'frames_axis_z_{i}', JIDX, pin.SE3(Z, position), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/Z_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl",np.array([0.00025,0.00025,0.00025]) ))
        # visual_model.geometryObjects[-1].meshColor = np.array([0, 0, 1, 1.0])
        
        # #desired frames pose
     
        # visual_model.addGeometryObject(pin.GeometryObject(f'df_axis_x_{i}', 0, pin.SE3(X, position), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/X_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/X_arrow.stl",np.array([0.0015,0.0015,0.0015]) ))
        # visual_model.geometryObjects[-1].meshColor = np.array([1, 0, 0, 1.0])

        # visual_model.addGeometryObject(pin.GeometryObject(f'df_axis_y_{i}', 0, pin.SE3(Y, position), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl",np.array([0.0015,0.0015,0.0015]) ))
        # visual_model.geometryObjects[-1].meshColor = np.array([0, 1, 0, 1.0])

        # visual_model.addGeometryObject(pin.GeometryObject(f'df_axis_z_{i}', 0, pin.SE3(Z, position), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/Z_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl",np.array([0.0015,0.0015,0.0015]) ))
        # visual_model.geometryObjects[-1].meshColor = np.array([0, 0, 1, 1.0])
    
    return visual_model



def traj_cam_linear(start, end, T):
    traj = [np.linspace(start[i], end[i], T) for i in range(3)]
    traj = np.array(traj).T
    return traj


def record_trajectory(
    viz, start_pos, qs, path="video.mp4", DT=0.015, fixed_cam=False, end_pos=None
):
    import imageio

    if end_pos is None and not fixed_cam:
        base_pos_diff = qs[-1, :3] - qs[0, :3]
        end_pos = start_pos + base_pos_diff
    if not fixed_cam:
        traj = traj_cam_linear(start_pos, end_pos, qs.shape[0])
    images = []
    for t, q in enumerate(qs):
        viz.display(q)
        if not fixed_cam:
            viz.setCameraPosition(traj[t])
        images.append(viz.viewer.get_image())
    imageio.mimsave(path, images, fps=1 // DT)



def box_between_frames(viz, name, T_c0, T_c2,
                    thickness=0.02, height=0.02,
                    rgba=(0.2, 0.7, 0.9, 0.8)):
    """
    Draw a rectangular box connecting the origins of frames c0 and c2.
    The box's local X axis is aligned with the segment (p2 - p1).

    viz   : Pinocchio MeshcatVisualizer
    name  : meshcat node name (string)
    T_c0  : 4x4 world->c0 transform
    T_c2  : 4x4 world->c2 transform
    thickness, height : box cross-section (meters)
    rgba  : color with alpha
    """
    p1 = np.asarray(T_c0[:3, 3], dtype=float)
    p2 = np.asarray(T_c2[:3, 3], dtype=float)
    v  = p2 - p1
    L  = float(np.linalg.norm(v))
    if L < 1e-12:
        # Degenerate: frames coincide. Draw a tiny dot-sized box.
        L = 1e-6
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
    else:
        ex = v / L  # direction along the segment
        up = np.array([0.0, 0.0, 1.0])
        if abs(ex @ up) > 0.95:   # avoid near-parallel 'up'
            up = np.array([0.0, 1.0, 0.0])
        ey = np.cross(up, ex); ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)

    # Rotation with columns = object axes in world (X along segment)
    R = np.column_stack((ex, ey, ez))
    c = 0.5 * (p1 + p2)

    # Build world transform for the box
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = c
    T[2, 3]-=0.1
    # Create geometry (length along local X)
    L+=0.1
    addViewerBox(viz, name, L, thickness, height, rgba)

    # Place it
    viz.viewer[name].set_transform(T)
    
    
def set_tf(viz, name, T_world_obj):

    viz.viewer[name].set_transform(T_world_obj)
        
def draw_table(viz, T_world_table):
    """
    Build a simple table (top + 4 legs) and place it at T_world_table.
    Uses addViewerBox() for both MeshCat and Gepetto viewers.

    Parameters
    ----------
    viz : pin.visualize.MeshcatVisualizer or GepettoVisualizer
    T_world_table : (4,4) ndarray homogeneous transform (world_T_table)
    """
    # ---- table parameters (meters) ----
    L, W, T = .90, 1.80, 0.04      # tabletop length, width, thickness
    H       = 0.95                  # total height
    LEG     = 0.05                  # leg square cross-section
    INSET   = 0.05                  # leg inset from edges

    # ---- names (unique in the scene) ----
    top_name = "table_top"
    leg_names = [
        "table_leg_00", "table_leg_01",
        "table_leg_10", "table_leg_11",
    ]

    # ---- create geometries (via your helper) ----
    # tabletop (brown)
    addViewerBox(viz, top_name, L, W, T, rgba=[0.80, 0.60, 0.40, 1.0])
    # legs (slightly darker)
    for n in leg_names:
        addViewerBox(viz, n, LEG, LEG, H - T, rgba=[0.45, 0.45, 0.45, 1.0])

    # ---- local poses (relative to the table frame) ----
    def homog(R=np.eye(3), t=(0, 0, 0)):
        Tm = np.eye(4)
        Tm[:3, :3] = R
        Tm[:3,  3] = np.array(t, dtype=float)
        return Tm

    # tabletop center is at z = H - T/2
    T_local_top = homog(t=(0.0, 0.0, H - T/2))

    # leg centers
    xs = [+L/2 - INSET - LEG/2, -L/2 + INSET + LEG/2]
    ys = [+W/2 - INSET - LEG/2, -W/2 + INSET + LEG/2]
    z_leg = (H - T)/2
    T_local_legs = [
        homog(t=(xs[0], ys[0], z_leg)),
        homog(t=(xs[0], ys[1], z_leg)),
        homog(t=(xs[1], ys[0], z_leg)),
        homog(t=(xs[1], ys[1], z_leg)),
    ]

    # ---- apply world transforms ----
    set_tf(viz, top_name, T_world_table @ T_local_top)
    for n, Tl in zip(leg_names, T_local_legs):
        set_tf(viz, n, T_world_table @ Tl)
