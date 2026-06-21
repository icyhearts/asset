print("swizzle")
for m in range(8):
    line = ""
    for n in range(32):
        offset = m * 32 + n
        swizzle = offset ^ (((offset & 0b111000000)) >> 3)
        line += f"{swizzle:3d} "
    print(line)


print("8 per group")

for m in range(8):
    line = ""
    for n in range(32):
        if n % 8 == 0:
            offset = m * 32 + n
            swizzle = offset ^ (((offset & 0b111000000)) >> 3)
            line += f"{swizzle:3d} "
    print(line)


print("8 per group // 8")
for m in range(8):
    line = ""
    for n in range(32):
        if n % 8 == 0:
            offset = m * 32 + n
            swizzle = offset ^ (((offset & 0b111000000)) >> 3)
            line += f"{swizzle//8:3d} "
    print(line)

for m in range(4):
    line = ""
    for n in range(8):
            offset = m ^  n
            swizzle = m * 8 + offset
            line += f"{swizzle:3d} "
            if (n+1) % 4 == 0:
                line += f"\n"
    print(line)
