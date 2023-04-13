import json

def parse_json(jpath):
    with open(jpath, 'r') as f:
        lb = json.load(f)
    boxes = [s['points'] for s in lb['shapes'] if len(s['points']) == 4]
    labels = [s['label'] for s in lb['shapes'] if len(s['points']) == 4]
    texts = [s['text'] for s in lb['shapes'] if len(s['points']) == 4]
    additions = {'impath':lb['imagePath'], 'h': lb['imageHeight'], 'w':lb['imageWidth']}
    return boxes, texts, labels, additions

def max_left(bb):
    return min(bb[0], bb[2], bb[4], bb[6])

def max_right(bb):
    return max(bb[0], bb[2], bb[4], bb[6])

def row_bbs(bbs_classes_labels):
    bbs_classes_labels.sort(key=lambda x: max_left(x[0]))
    clusters, y_min, cluster_texts, cluster_labels = [], [], [], []
    for tgt_node, text, label in bbs_classes_labels:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            cluster_texts.append([text])
            cluster_labels.append([label])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            cluster_texts.append([text])
            cluster_labels.append([label])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
            cluster_texts[idx].append(text)
            cluster_labels[idx].append(label)
    zip_clusters = list(zip(clusters, y_min, cluster_texts, cluster_labels))
    zip_clusters.sort(key=lambda x: x[1])
    return zip_clusters

def sort_bbs(bbs, texts, labels):
    bbs_classes_labels = [(b, c, l) for b, c, l in zip(bbs, texts, labels)]
    bb_clusters = row_bbs(bbs_classes_labels)
    return bb_clusters