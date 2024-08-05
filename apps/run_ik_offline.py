from utils.model_utils import Robot

# Loading human urdf
human = Robot('models/human_urdf/urdf/human.urdf','models') 
human_model = human.model
human_visual_model = human.visual_model