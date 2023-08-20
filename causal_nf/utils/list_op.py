import re


def list_intersection(list1, list2):
    """
    Returns the intersection of two lists.
    """
    return list(set(list1) & set(list2))


def list_union(list1, list2):
    """
    Returns the intersection of two lists.
    """
    return list(set(list1) | set(list2))


def list_sub(list1, list2):
    """
    Returns the first list minus the elements in the second list.
    """
    return [item for item in list1 if item not in list2]


def list_selector(list_, regex_filter=".*", regex_exclude="^$"):
    elem_selected = []
    elem_hidden = []
    for elem in list_:
        if re.match(regex_filter, elem) and not re.match(regex_exclude, elem):
            elem_selected.append(elem)
        else:
            elem_hidden.append(elem)
    return elem_selected, elem_hidden


def show_list_selector(list_, regex_filter=".*", regex_exclude="^$"):
    elem_selected, elem_hidden = list_selector(
        list_=list_, regex_filter=regex_filter, regex_exclude=regex_exclude
    )

    for el in elem_selected:
        print(f"{el}")
