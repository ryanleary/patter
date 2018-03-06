def split_targets(targets, target_sizes):
    results = []
    offset = 0
    for size in target_sizes:
        results.append(targets[offset:offset + size])
        offset += size
    return results
