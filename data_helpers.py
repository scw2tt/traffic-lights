""""
Given a number in the TLD format,
this will output the number that corresponds
to the color traffic light in our repo's
format.

Returns:
    int -- number corresponding to the state (color)
    of the traffic light
"""


def get_state(class_id):
    state_id = class_id[4]

    # red is zero
    if state_id == "1":
        return 0

    # yellow is one
    elif state_id == "2":
        return 1

    # green is two
    elif state_id == "4":
        return 2

    # else is negative 1 and should be removed from the dataset
    else:
        return -1


"""
This function will allow us to convert the long integer formatted class into
the state of the traffic light for the current image. This is conveyed in the
5th digit of the class_id. It also returns the path to get the corresponding
image.
"""


def get_label_and_path(file_path, index, images):
    # out_labels contains
    out_states = []
    out_bounding_boxes = []
    out_path = None

    # get the image from the file
    image_dict = images[index]['objects']
    out_path = images[index]['path']

    for o in image_dict:
        # turn the class_id into the output color
        curr_state = get_state(str(o['class_id']))
        curr_b_box = [o['x'], o['y'], o['width'], o['height']]

        # save bounding box and data to the output to the output
        out_states.append(curr_state)
        out_bounding_boxes.append(curr_b_box)
    # return the output
    return out_path, out_bounding_boxes, out_states
