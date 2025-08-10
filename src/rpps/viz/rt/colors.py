import matplotlib.colors as colors

hot = colors.LinearSegmentedColormap.from_list("RT-hot", (
    (0.00, (0.0, 0.0, 0.0)),
    (0.05, (0.0, 0.0, 0.2)),
    (0.10, (0.0, 0.0, 0.5)),
    (0.50, (0.0, 0.8, 0.0)),
    (1.00, (1.0, 0.0, 0.0)),
))

cold = colors.LinearSegmentedColormap.from_list("RT-cold", (
    (0.00, (0.0, 0.0, 0.0)),
    (0.05, (0.1, 0.0, 0.0)),
    (0.10, (0.4, 0.0, 0.0)),
    (0.50, (0.0, 0.8, 0.0)),
    (1.00, (0.0, 0.0, 1.0)),
))

radar = colors.LinearSegmentedColormap.from_list("RT-radar", (
    (0.00, (0.0, 0.0, 0.0)),
    (0.05, (0.0, 0.1, 0.0)),
    (0.50, (0.2, 0.8, 0.2)),
    (1.00, (1.0, 1.0, 1.0)),
))

gray = colors.LinearSegmentedColormap.from_list("RT-radar", (
    (0.00, (0.0, 0.0, 0.0)),
    (0.50, (0.5, 0.5, 0.5)),
    (1.00, (1.0, 1.0, 1.0)),
))

s = [k for k in globals().keys() if not k.startswith("_") and not k == "colors"]

def get(name):
    return globals()[name]
