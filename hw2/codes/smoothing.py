
group_map = {
    'aa': 1,
    'ae': 1,
    'ah': 1,
    'aw': 1,
    'ay': 1,
    
    'dx': 2,
    't': 2,

    'm': 3,
    'n': 3,
    'ng': 3,

    'r': 4,
    'er': 4,

    'ow': 5,
    'oy': 5,
}


def smo1(s):

    res = []
    i = 0

    while i < len(s):

        j = i
        while j+1 < len(s) and group_map.get(s[j+1], 0) != 0 \
                and group_map.get(s[i], 0) == group_map.get(s[j+1], 0):

            j += 1
        
        res.append(s[i])
        i = j + 1

    return res


