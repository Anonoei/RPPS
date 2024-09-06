def main():
    points = [
        0.7 - 0.7j, -0.7 - 0.7j,
        0.7 + 0.7j, -0.7 + 0.7j
    ]
    mapping = [
        0, 1,
        2, 3,
    ]

    inv = [num.real + (-num.imag) * 1j for num in points]
    inv_swap = inv[::2][::-1] + inv[::-1][::2]

    print(f"Normal: {[mapping[points.index(p)] for p in points]}")
    print(f"Invert: {[mapping[points.index(p)] for p in inv]}")
    print(f"I-Swap: {[mapping[points.index(p)] for p in inv_swap]}")


if __name__ == "__main__":
    main()
