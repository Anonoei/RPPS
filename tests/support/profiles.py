"""What the profile tests share"""
def names(runner):
    return [f"{how}{type(op).__name__}" for how, op in runner]
