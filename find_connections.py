from collections import defaultdict
import numpy as np
import yaml
import json

with open('standard_lego_library.yaml', 'r') as file:
    lego_lib = yaml.safe_load(file)
file.close()


def create_block_graph(assembly_list):
    graph = defaultdict(list)

    for node1 in assembly_list:
        piece = assembly_list[node1]
        id1 = piece['brick_id']
        x1, y1, z1 = piece['x'], piece['y'], piece['z']
        ori1 = piece['ori']

        for node2 in assembly_list:
            piece2 = assembly_list[node2]
            id2 = piece2['brick_id']
            x2, y2, z2 = piece2['x'], piece2['y'], piece2['z']
            ori2 = piece2['ori']

            if node1 != node2:
                if is_touching(piece, piece2):
                    graph[node1].append(node2)
    return graph


def is_touching(piece, piece2):
    height1 = lego_lib[piece['brick_id']]['height']
    width1 = lego_lib[piece['brick_id']]['width']
    depth1 = 1  # Sudo Value
    height2 = lego_lib[piece2['brick_id']]['height']
    width2 = lego_lib[piece2['brick_id']]['width']
    depth2 = 1  # Sudo Value

    x1, y1, z1, ori1 = piece['x'], piece['y'], piece['z'], piece['ori']
    x2, y2, z2, ori2 = piece2['x'], piece2['y'], piece2['z'], piece2['ori']

    if ori1 == 0:
        width1, height1 = height1, width1
    if ori2 == 0:
        width2, height2 = height2, width2

    if ((x1 + width1 == x2 and (y1 < y2 + height2 and y1 + height1 > y2) and z1 == z2) or
            (x2 + width2 == x1 and (y2 < y1 + height1 and y2 + height2 > y1) and z1 == z2) or
            (y1 + height1 == y2 and (x1 < x2 + width2 and x1 + width1 > x2) and z1 == z2) or
            (y2 + height2 == y1 and (x2 < x1 + width1 and x2 + width2 > x1) and z1 == z2) or
            (z1 + depth1 == z2 and (x1 < x2 + width2 and x1 + width1 > x2) and (
                    y1 < y2 + height2 and y1 + height1 > y2)) or
            (z2 + depth2 == z1 and (x2 < x1 + width1 and x2 + width2 > x1) and (
                    y2 < y1 + height1 and y2 + height2 > y1))):
        return True
    return False


if __name__ == '__main__':
    board_size = [48, 48]  # Specify the depth as 1
    f = open("../config/assembly_tasks/chair.json")
    assembly_list = json.load(f)

    # with open('example_list2.yaml', 'r') as file:
    #     assembly_list = yaml.safe_load(file)
    # file.close()

    with open('standard_lego_library.yaml', 'r') as file:
        lego_lib = yaml.safe_load(file)
    file.close()

    for p in assembly_list:
        print(p + ": " + str(assembly_list[p]))

    graph = create_block_graph(assembly_list)
    print(graph)

    json_object = json.dumps(graph, indent=4)

    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)
