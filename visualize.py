import json
import matplotlib
import matplotlib.pyplot as plt
import yaml

matplotlib.use('TkAgg')


def display_block_graph(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for p in data:
        piece = data[p]
        piece_id = piece['brick_id']
        x = piece['x']
        y = piece['y']
        z = piece['z']
        height = lego_lib[piece_id]['height']
        width = lego_lib[piece_id]['width']
        depth = 1  # Sudo value
        color = lego_lib[piece_id]['color']
        orientation = piece['ori']

        if orientation == 0:
            width, height = height, width

        x_start = x
        y_start = y
        z_start = z

        if color == "cream": color = "beige"
        ax.bar3d(x_start, y_start, z_start, width, height, depth, color, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f.name.title())

    # Adjust viewing angle
    ax.view_init(elev=60, azim=90)

    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()


f = open("C:/Users/Alan/PycharmProjects/AssemblyCompletion/lego_structures/chair_simple/save/it10000-export/task_graph.json")
assembly_list = json.load(f)

with open('standard_lego_library.yaml', 'r') as file:
    lego_lib = yaml.safe_load(file)
file.close()

display_block_graph(assembly_list)
