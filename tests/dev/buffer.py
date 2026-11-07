import rpps as rp

import numpy as np

def main():
    # test_buffer()
    test_vec()
    # test_processor()

def test_buffer():
    b1 = rp.base.buffer.Buffer(10)
    print(f"Init {str(b1)}")
    n = np.arange(0,21)
    print(f"Set 1 * {len(n)}")
    for i in n:
        b1.set(i)
        data = b1.get(10)
        if not data is None:
            print(f"  got: {data}")
    b1.reset()
    print(f"Set {len(n)}")
    b1.set(n)
    print(f"  got: {b1.get(10)}")
    print(f"  {str(b1)}")

    print("Resize 20")
    b1.resize(20)
    print(f"  {str(b1)}")

    print("Resize 5")
    b1.resize(5)
    print(f"  {str(b1)}")

    print("Resize 20")
    b1.resize(20)
    print(f"  {str(b1)}")

def test_vec():
    v1 = rp.base.buffer.Vector()
    print(f"Init {str(v1)}")
    n = np.arange(0,21)
    print(f"Set 1 * {len(n)}")
    for i in n:
        v1.set(i)
        data = v1.get(10)
        if not data is None:
            print(f"  got: {data}")
    print(f"{str(v1)}")

def test_processor():
    p1 = rp.base.buffer.Processor(10)
    n = np.arange(0,21)
    for i in n:
        p1.encode(i)
    print(p1)



if __name__ == "__main__":
    main()
