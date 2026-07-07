import torch
from MaskClustering.utils import Node


def get_connected_components(graph):
    visited = set()
    components = []

    for node in graph.keys():
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        component = []
        while stack:
            cur = stack.pop()
            component.append(cur)
            for neigh in graph[cur]:
                if neigh not in visited:
                    visited.add(neigh)
                    stack.append(neigh)

        components.append(component)

    return components

def components_to_nodes(
    components,
    global_frame_mask_list,
    mask_point_clouds,
    frame_list,
):
    node_list = []
    for cluster_id, component in enumerate(components):
        mask_list = []
        point_ids = set()
        visible_frame = torch.zeros(len(frame_list), dtype=torch.bool)
        
        for global_id in component:
            frame_id, mask_id = global_frame_mask_list[global_id]
            mask_list.append((frame_id, mask_id))
            point_ids.update(mask_point_clouds[f"{frame_id}_{mask_id}"])
            frame_index = frame_list.index(frame_id)
            visible_frame[frame_index] = True

        if len(mask_list) < 2:
            continue
        node = Node(
            mask_list=mask_list,
            visible_frame=visible_frame,
            contained_mask=None,
            point_ids=point_ids,
            node_info=(0, cluster_id),
            son_node_info=None,
        )
        node_list.append(node)

    return node_list