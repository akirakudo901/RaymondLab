# under bsoid_app/bsoid_utilities/bsoid_classification bsoid_predict
def bsoid_predict(feats, clf):
    """
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(0, len(feats)):
        labels = clf.predict(feats[i].T)
        labels_fslow.append(labels)
    return labels_fslow