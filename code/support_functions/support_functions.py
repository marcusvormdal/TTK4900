import numpy as np

def get_relative_pos(object, obj_t = 'point', centered = False):

    if obj_t == 'point':
        if np.size(object) == 0:
            return np.array([])
        new_obj = np.empty_like(object)
        new_obj = (object / 5)- 10
        
        return new_obj
    
    elif obj_t == 'line':
        relative_lines = []
        if object[0] == None:
            return []
        for line in object[0]:
            x0 = (line[0][0] / 5)- 10
            y0 = (line[0][1] / 5)- 10
            x1 = (line[0][2] / 5)- 10
            y1 = (line[0][3] / 5) -10
            rel_line = [(y0, y1),(x0, x1)]

            relative_lines.append(rel_line)           
        return relative_lines
    
    return None

def get_image_pos(elem):
    image_elem = (np.floor((elem+10)*5)).astype(int)
    
    return image_elem

def get_position():
    
    return 0,0,0

def get_orientation():
    
    return(0,0,0)


def update_orientation(new_orientation, element):
    
    # T @ v
    return