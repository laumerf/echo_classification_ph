#
# Helper functions for label creation, and label dictionary definitions for various versions of labels.
#


def get_legal_float_labels(raw_ph_label):
    """
    Given a raw label, return empty string if not legal label (e.g. nan, 'undecided', or wrong range).
    In case of a legal label, return the floating point equivalent of the label (0, 0.5, 1, 1.5, 2, 2.5, 3).
    (The .5 comes from labels 'between' two categories.)
    :param raw_ph_label:
    :return: Floating point mapping of the label, or 0.0 if non-legal.
    """
    if isinstance(raw_ph_label, int) and 0 <= raw_ph_label <= 3:  # legal
        return float(raw_ph_label)
    if isinstance(raw_ph_label, str):  # string is legal if contains 'bis'
        if 'bis' in raw_ph_label:
            return (int(raw_ph_label[-1]) + int(raw_ph_label[0]))/2.0
        else:  # Not a legal label (e.g. 'nichts bestimmt', etc.)
            return -1
    return -1   # if the label is not int, nor string, e.g. a nan - then not legal


# Mapping the original four labels (including in-between labels) to 3 classes.
# Keys represent original labels, values new labels.
# .5 keys represent labels in-between two original labels.
label_map_3class = ({0: 0, 0.5: None, 1: 1, 1.5: 1, 2: 2, 2.5: 2, 3: 2},
                    '3class')  # later parameter is descriptive name

# Try to disregard the extreme case (3)
label_map_3class_2 = ({0: 0,
                       0.5: None,
                       1: 1,
                       1.5: 1,
                       2: 2,
                       2.5: 2,
                       3: None}, '3class_2')

label_map_3class_3 = ({0: 0,
                       0.5: None,
                       1: 1,
                       1.5: 2,
                       2: 2,
                       2.5: 2,
                       3: None}, '3class_3')

label_map_4class = ({0: 0, 0.5: None, 1: 1, 1.5: 2, 2: 2, 2.5: 3, 3: 3}, '4class')


# Mapping the original four labels (including in-between labels) to 2 classes.
# Keys represent original labels, values new labels.
# .5 keys represent labels in-between two original labels.
# 0 and 0-1 is 'normal', rest (1, 1-2, 2, 2-3, 3) is 'abnormal)
label_map_2class = ({0: 0, 0.5: 0, 1: 1, 1.5: 1, 2: 1, 2.5: 1, 3: 1},
                    '2class')

# Mapping the original four labels (including in-between labels) to 2 classes.
# Keys represent original labels, values new labels.
# .5 keys represent labels in-between two original labels.
label_map_2class_drop0bis1_drop3 = ({0: 0, 0.5: None, 1: 1, 1.5: 1, 2: 1, 2.5: 1, 3: None},
                                    '2class_drop_ambiguous')  # later parameter is descriptive name

long_label_type_to_short = {'3class': '3',
                            '3class_2': '3_2',
                            '3class_3': '3_3',
                            '4class': '4',
                            '2class': '2',
                            '2class_drop_ambiguous': '2d'}

