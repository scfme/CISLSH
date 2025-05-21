import warnings
warnings.filterwarnings("ignore")
import numpy as np

def CISLSH(Data, label, W, M, L, eta):

    num_samples, Dim = Data.shape
    a = np.random.normal(0, 1, [M * L, Dim])
    b = W * np.random.rand(M * L, 1)
    select_data = []
    matrix = []

    for i in range(L):
        j = np.arange(i * M, (i + 1) * M)
        bucket_index = np.floor((np.dot(a[j, :], Data.T) + b[j, :]) / W) % 2
        transposed_matrix = bucket_index.T
        binary_strings = [''.join(map(str, map(int, col))) for col in transposed_matrix]
        decimal_values = [int(binary_string, 2) for binary_string in binary_strings]
        matrix.append(decimal_values)
    bucket_index_matrix = np.array(matrix,dtype=np.int32)
    # -------------------------BIM完成---------------------"#

    vote_matrix = np.zeros((num_samples, num_samples), dtype=np.int32)

    for layer in range(L):
        layer_buckets = bucket_index_matrix[layer]
        same_bucket = layer_buckets[:, np.newaxis] == layer_buckets
        vote_matrix += same_bucket

    non_zero_elements = vote_matrix[vote_matrix != 0]
    average_non_zero = np.mean(non_zero_elements)
    # -------------------------VM完成-------------------------#

    final_buckets = {}
    unassigned_samples = set(range(num_samples))
    current_bucket = 0

    while unassigned_samples:
        current_bucket += 1
        final_buckets[current_bucket] = []
        sample = unassigned_samples.pop()
        final_buckets[current_bucket].append(sample)

        for other_sample in list(unassigned_samples):
            if vote_matrix[sample, other_sample] >= average_non_zero + eta:
                final_buckets[current_bucket].append(other_sample)
                unassigned_samples.remove(other_sample)
    # -------------------------分桶完成-------------------------#

    for value, indices in final_buckets.items():
        center_bucket = np.mean(Data[indices], axis=0).astype(np.float64)
        distances = np.linalg.norm(Data[indices] - center_bucket, axis=1).astype(np.float64)

        if np.max(distances) >= 0.005:
            min_distance_ind = np.argmin(distances)
            class_indices = indices[min_distance_ind]
            select_data.append(class_indices)
    # "-------------------------选择完成----------------------"#

    select_data_end = list(set(select_data))
    select_sub_x, select_label_y = Data[select_data_end], label[select_data_end]
    return select_sub_x, select_label_y, final_buckets
