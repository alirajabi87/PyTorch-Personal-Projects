with open("../../Data/coco/trainvalno5k.txt", 'r') as file:
    names = file.read().split('\n')
    names = names[:-1]
    new_names = []
    for i in range(len(names)):
        parts = names[i].split("_")
        parts[-1] = parts[-1][1:]
        new_names.append("_".join(parts))

    print(names[1])
    print(new_names[1])