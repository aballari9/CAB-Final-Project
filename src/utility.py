import numpy as np

def conv_list_representation(mat_representation):
    tmp1 = {}
    for row in mat_representation:
        if not tmp1.get(row[2]):
            tmp1[row[2]] = []
        tmp1[row[2]].append((row[0], row[1]))
    return list(tmp1.values())

def normalize_list_representation(list_representation):
    new_output = []
    for item in list_representation:
        new_cluster = []
        for point in item:
            if isinstance(point, dict):
                new_cluster.append((point["x"], point["y"]))
            else:
                assert isinstance(point, tuple)
                assert len(point) == 2
                new_cluster.append(point)
        new_output.append(new_cluster)
    return new_output

def conv_mat_representation(list_representation):
    rows = []
    for cluster_idx in range(0, len(list_representation)):
        for point in list_representation[cluster_idx]:
            if isinstance(point, dict):
                rows.append((point["x"], point["y"], cluster_idx))
            else:
                # probably is a tuple here
                rows.append((*point, cluster_idx))

    return np.array(rows)

def calc_fowlkes_mallows(
    input_clusters_1, input_clusters_2
):
    """
    Calculate Fowlkes-Mallows indices for the clusters.
    """

    # pylint: disable=invalid-name

    clusters_1 = (conv_list_representation(input_clusters_1)
                  if isinstance(input_clusters_1, np.ndarray) else normalize_list_representation(input_clusters_1))
    clusters_2 = (conv_list_representation(input_clusters_2)
                  if isinstance(input_clusters_2, np.ndarray) else normalize_list_representation(input_clusters_2))

    all_points = set(point for cluster in clusters_1 for point in cluster)

    # Using frozenset for hashability and order independence
    pairs_of_points = set(
        frozenset((point1, point2))
        for point1 in all_points
        for point2 in all_points
        if point1 != point2
    )

    # make clusters into sets of points for member testing
    clusters_1 = [set(x) for x in clusters_1]
    clusters_2 = [set(x) for x in clusters_2]


    # for fowlkes-mallows
    tp = 0
    fp = 0
    fn = 0

    for pair in pairs_of_points:

        same1 = False
        same2 = False

        for cluster in clusters_1:
            if pair.issubset(cluster):
                same1 = True
                break

        for cluster in clusters_2:
            if pair.issubset(cluster):
                same2 = True
                break

        if same1 and same2:
            tp += 1
        if same1 and not same2:
            fp += 1
        if not same1 and same2:
            fn += 1

    try:
        fowlkes_mallows = ((tp / (tp + fp)) * (tp / (tp + fn))) ** 0.5
        return fowlkes_mallows
    except ZeroDivisionError:
        return 0
        # print(input_clusters_1)
        # print(input_clusters_2)
        # print(clusters_1)
        # print(clusters_2)
        # print(tp, fp, fn)
        # raise e

