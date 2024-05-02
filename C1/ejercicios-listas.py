test_list = [1, 2, 3, 4, 5, 2, 3]
test_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
test_set2 = (1, 2, 3, 4)


def even_list(lst):
    new_list = []
    for element in lst:
        if element % 2 == 0:
            new_list.append(element)
    print(f"Solo pares: {new_list}")
    return new_list


def sum_and_mean(lst):
    if not lst:
        return 0, 0

    summ = 0
    for element in lst:
        summ += element

    mean = summ / len(lst)

    print(f"Suma: {summ}\nMean: {mean}")
    return summ, mean


def drop_duplicated(lst):
    unique_list = set(lst)
    sorted_list = sorted(unique_list)

    print(f"Sin duplicados: {sorted_list}")
    return sorted_list


def get_event_elements(elements):
    even_elements = [elem for index, elem in enumerate(elements) if index % 2 == 0]
    print(f"Elementos en posiciones pares: {even_elements}")
    return even_elements


def get_concat_set(set1, set2):
    new_set = set1 + set2
    print(f"Tupla concatenada: {new_set}")
    return new_set


def get_elements_type(elements):
    str_count = 0
    int_count = 0
    for element in elements:
        if type(element) is str:
            str_count += 1
        elif type(element) is int:
            int_count += 1
    print(f"String count: {str_count}\nInt Count: {int_count}")
    return str_count, int_count


even_list(test_list)
sum_and_mean(test_list)
drop_duplicated(test_list)

print("\n" + "-" * 20 + "\n")

get_event_elements(test_set)
get_concat_set(test_set, test_set2)
get_elements_type(test_set)
