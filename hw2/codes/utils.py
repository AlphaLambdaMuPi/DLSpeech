
def psi(dt, mp):
    PHONES = 48
    FBANKS = 69
    DIMS = PHONES * FBANKS + PHONES * PHONES
    res = [0.0] * DIMS

    def label_id(l):
        return mp[l][0] 
    def label_feature_id(l, f):
        return l * FBANKS + f
    def trans_id(a, b):
        return PHONES * FBANKS + a * PHONES + b

    __, fe, la = zip(*dt)
    last_lid = -1
    for i in range(len(__)):
        lid = label_id(la[i])

        for j in range(FBANKS):
            res[label_feature_id(lid, j)] += fe[i][j]

        if last_lid != -1:
            res[trans_id(last_lid, lid)] += 1.0

        last_lid = lid

    return res
