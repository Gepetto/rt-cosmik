import numpy as np
import tomli
import pinocchio as pin
from utils.parameters_class import Parameters

def evaluate_str(expr: str) -> float:
    # Replace 'pi' with the actual value
    expr = expr.replace('pi', f'({np.pi})')
    # Evaluate safely with no built-ins
    return eval(expr, {"__builtins__": {}}, {"np": np})

def parse_type_dict(current_dict: dict):
    assert "type" in current_dict.keys()
    
    t = current_dict["type"]

    ##################
    ## Parse Arrays ##
    ##################
    if t == "array":
        assert "size" in current_dict.keys()
        size = current_dict["size"]

        ### Fill array with given value
        if "fill" in current_dict.keys():
            fill = current_dict["fill"]
            if type(fill) in [int, float]:
                fill = float(fill)
            else:
                assert type(fill) == str
                fill = float(evaluate_str(fill))  # Interpret the expression
        else:
            fill = 0.0

        ### Convert the fill value if needed
        if "deg2rad" in current_dict.keys() and current_dict["deg2rad"]:
            arr = np.ones(size) * np.deg2rad(fill)
        else:
            arr = np.ones(size) * fill

        return arr

    ###############
    ## Parse SE3 ##
    ###############
    elif t == "SE3":
        assert "rotation" in current_dict.keys() and "position" in current_dict.keys()

        ## Parse the position
        assert type(current_dict["position"]) is list and len(current_dict["position"]) == 3
        pos = np.array(current_dict["position"])

        ## Parse the rotation
        rot = current_dict["rotation"]
        if type(rot) is str:
            if rot.lower() == "identity":
                rot = np.eye(3)
            else:
                raise NotImplementedError(
                    "Rotation values for SE3 can only be 'identity' or a list representing a quaternion or a rotation matrix"
                )
        elif type(rot) is list:
            assert np.shape(rot) in [(4,), (3, 3)]
        else:
            raise NotImplementedError(
                "Rotation values for SE3 should be of size (4,) or (3,3)"
            )

        return pin.SE3(rot, pos)

    else:
        raise NotImplementedError(f"Type '{t}' not supported")
        #case _:
        #    raise NotImplementedError(f"In dict {main_key}, type {current_dict["type"]} is not supported")

def parse_raw_params(raw_params: dict, depth: int = 0) -> dict:
    print(depth)
    if depth > 1:
        raise RecursionError("Too much depth has been found in the parameters, please consider splitting the parameters")
    params_dict = {}
    for main_key in raw_params.keys():
        # Iterate over the main keys
        main_value = raw_params[main_key]
        assert type(main_value) in [float, int, bool, dict, str, list]
        if type(main_value) in [float, int, bool, str, list]:
            params_dict[main_key] = main_value
        else: # i.e. if it is a dict
            if "type" in main_value.keys():
                params_dict[main_key] = parse_type_dict(main_value)
            else:
                # Then allow one recursion
                params_dict[main_key] = parse_raw_params(main_value, depth=depth+1)                
    return params_dict

def parse_params(file_name: str) -> dict:
    print(file_name)
    with open(file_name, "rb") as f:
        print(f)
        raw_params = tomli.load(f)
    return(parse_raw_params(raw_params))

def convert_to_class(dict):
    keys = Parameters.__dataclass_fields__.keys() & dict.keys()
    kwargs = {key: dict[key] for key in keys}
    param = Parameters(**kwargs)
    return param

#dict_param = parse_params("./parameters.tomli")
#class_param = convert_to_class(dict_param)

 