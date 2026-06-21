for m in range(8):
    line = ""
    for n in range(32):
        offset = m * 32 + n
        swizzle = offset ^ (((offset & 0b111000000)) >> 3)
        line += f"{swizzle:3d} "
    print(line)
