def get_tests(glbs):
    tests = []
    for k, v in glbs.items():
        if k.startswith("test"):
            tests.append(v)
    return tests


def run_tests(glbs):
    for test in get_tests(glbs):
        test()
