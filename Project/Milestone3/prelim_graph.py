

from MaskClustering.graph.construction import build_point_in_mask_matrix, get_observer_num_thresholds, init_nodes, process_masks
from MaskClustering.graph.node import Node


def init_nodes_edges(
    global_frame_mask_list,
    mask_point_clouds,
    undersegment_mask_ids,
    boundary_points,
    boundary_overlap_threshold=0.3,
    min_boundary_overlap_points=20,
):
    valid_global_ids = [
        i for i in range(len(global_frame_mask_list))
        if i not in undersegment_mask_ids
    ]

    nodes = {}
    for global_id in valid_global_ids:
        frame_i, mask_i = global_frame_mask_list[global_id]
        points_i = mask_point_clouds[f"{frame_i}_{mask_i}"]
        nodes[global_id] = Node(mask_list=[(frame_i, mask_i)],
                      visible_frame=None,
                      contained_mask=None,
                      point_ids=points_i,
                      node_info=(0, global_id),
                      son_node_info=None)
    

    edges = {}
    for a in range(len(valid_global_ids)):
        global_i = valid_global_ids[a]
        frame_i, mask_i = global_frame_mask_list[global_i]
        node_i = nodes[global_i]

        points_i = mask_point_clouds[f"{frame_i}_{mask_i}"]
        boundary_i = points_i.intersection(boundary_points)

        if len(boundary_i) == 0:
            continue

        for b in range(a + 1, len(valid_global_ids)):
            global_j = valid_global_ids[b]
            frame_j, mask_j = global_frame_mask_list[global_j]
            node_j = nodes[global_j]

            if frame_i == frame_j: continue
            points_j = mask_point_clouds[f"{frame_j}_{mask_j}"]
            boundary_j = points_j.intersection(boundary_points)
            if len(boundary_j) == 0: continue
            inter = boundary_i.intersection(boundary_j)
            inter_size = len(inter)
            if inter_size < min_boundary_overlap_points: continue
            #TODO: consider finding the true geomteric edge points by finding points
            #with high surface variation to curvature ratio within k local neighborhood.
            boundary_overlap = inter_size / min(len(boundary_i), len(boundary_j))
            if boundary_overlap >= boundary_overlap_threshold:
                edges[(node_i, node_j)] = boundary_overlap

    return nodes, edges

def build_graph(nodes, edges, threshold=0.3):
    graph = {}
    for node in nodes.values():
        graph[node] = []
    for (node_i, node_j), weight in edges.items():
        if weight >= threshold:
            graph[node_i].append(node_j)
            graph[node_j].append(node_i)
    return graph

def prelim_graph(args, scene_points, frame_list, dataset):
    if args.debug:
        print('start building point in mask matrix')

    boundary_points, point_in_mask_matrix, mask_point_clouds, point_frame_matrix, global_frame_mask_list = \
        build_point_in_mask_matrix(args, scene_points, frame_list, dataset)

    _, __, undersegment_mask_ids = process_masks(
        frame_list,
        global_frame_mask_list,
        point_in_mask_matrix,
        boundary_points,
        mask_point_clouds,
        args
    )

    nodes, preliminary_edges = init_nodes_edges(
        global_frame_mask_list,
        mask_point_clouds,
        undersegment_mask_ids,
        boundary_points,
        boundary_overlap_threshold=args.prelim_overlap_threshold,
        min_boundary_overlap_points=args.prelim_min_overlap_points,
    )

    graph = build_graph(nodes,preliminary_edges,threshold=args.prelim_overlap_threshold)

    return graph, mask_point_clouds, point_frame_matrix, global_frame_mask_list