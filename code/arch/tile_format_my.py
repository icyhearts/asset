def tile_shape_config(global_dim: list, # in number of elements
                      datatype_bit: int,  # 32/16/8/6/4
                      is_mx: bool = True,
                      ):
    subtile_dim = [0] * 2  # Sub-Tile shape [K, M], in elements
    tile_dim = [0] * 2  # Tile Shape [K, M], in elements
    supertile_dim = [0] * 2  # Tile shape [K, M], in tiles
    tile_padding_elements = [0] * 2  # tile-level padding
    supertile_padding_tiles = [0] * 2  # supertile-level padding

    if datatype_bit == 6:
        num_subtile_in_tile = 8
        subtile_data_byte = 384
        subtile_dim = [64, 8]
        tile_data_byte = num_subtile_in_tile * subtile_data_byte  # only for data. Header byte not included
    else:
        num_subtile_in_tile = 4
        subtile_data_byte = 256
        subtile_dim = [int(subtile_data_byte // 8 // (float(datatype_bit) / 8)), 8]
        tile_data_byte = num_subtile_in_tile * subtile_data_byte  # only for data. Header byte not included

    print("*** global_dim [K, M] = {}, datatype_bit = {} ***".format(global_dim, datatype_bit))
    assert(len(global_dim) >= 2), "Tensor Global Dim must be equal or above two."

    if global_dim[1] > 16:
        # Tile Shape: M32
        tile_dim[1] = 32  # M dim
        tile_dim[0] = int(tile_data_byte // tile_dim[1] // (float(datatype_bit) / 8))  # K dim # in element: tile_data_byte // tile_dim[1]= bytes per row, (float(datatype_bit) / 8)=bytes per element, int(tile_data_byte // tile_dim[1] // (float(datatype_bit) / 8): num elements per row
        if is_mx == False:
            supertile_dim[0] = 1
            supertile_dim[1] = 1
        else:
            if global_dim[0] > 2 * tile_dim[0]:
                supertile_dim[0] = 4
                supertile_dim[1] = 1
            elif global_dim[0] <= 2 * tile_dim[0] and global_dim[0] > 1 * tile_dim[0]:
                supertile_dim[0] = 2
                supertile_dim[1] = 2
            elif global_dim[0] <= 1 * tile_dim[0] and global_dim[0] > 0:
                supertile_dim[0] = 1
                supertile_dim[1] = 4
            else:
                print("global_dim {} invalid".format(global_dim))
    elif global_dim[1] <= 16 and global_dim[1] > 8:
        # Tile Shape: M16
        tile_dim[1] = 16  # M dim
        tile_dim[0] = int(tile_data_byte // tile_dim[1] // (float(datatype_bit) / 8))  # K dim
        if is_mx == False:
            supertile_dim[0] = 1
            supertile_dim[1] = 1
        else:
            supertile_dim[0] = 4
            supertile_dim[1] = 1
    elif global_dim[1] <= 8 and global_dim[1] > 1:
        # Tile Shape: M8
        tile_dim[1] = 8  # M dim
        tile_dim[0] = int(tile_data_byte // tile_dim[1] // (float(datatype_bit) / 8))  # K dim
        if is_mx == False:
            supertile_dim[0] = 1
            supertile_dim[1] = 1

        else:
            supertile_dim[0] = 4
            supertile_dim[1] = 1
    elif global_dim[1] == 1:
        # Using vector format
        tile_dim[1] = 1  # M dim
        tile_dim[0] = int(tile_data_byte // tile_dim[1] // (float(datatype_bit) / 8))  # K dim
        if is_mx == False:
            supertile_dim[0] = 1
            supertile_dim[1] = 1
        else:
            supertile_dim[0] = 4
            supertile_dim[1] = 1
    else:
        print("global_dim {} invalid".format(global_dim))

    print("Subtile Dim (in elements, [K, M]) = {}".format(subtile_dim))
    print("Tile Dim (in elements, [K, M]) = {}".format(tile_dim))
    print("SuperTile Dim (in tiles, [K, M]) = {}".format(supertile_dim))

    # tile-level padding
    for dim in range(2):
        if global_dim[dim] % tile_dim[dim] != 0:
            tile_padding_elements[dim] = tile_dim[dim] - global_dim[dim] % tile_dim[dim]
            print("tile_level padding: dim {} pad {} elements".format(dim, tile_padding_elements[dim]))

    # supertile-level padding

    for dim in range(2):
        global_dim_tiles = (global_dim[dim] + tile_padding_elements[dim]) // tile_dim[dim] # global_dim_in_elements_after_above_padd // tile_dim = num tile after padding tile
        if global_dim_tiles % supertile_dim[dim] != 0: # supertile_dim 的单位是: number of tile,一个tile = 1
            supertile_padding_tiles[dim] = supertile_dim[dim] - global_dim_tiles % supertile_dim[dim]
            print("supertile_level padding: dim {} pad {} tiles".format(dim, supertile_padding_tiles[dim]))


    print("\n\n")


def main():
    tile_shape_config([1000,1024], 4)
    tile_shape_config([1000,32], 4)
    tile_shape_config([1000,16], 4)
    tile_shape_config([1000,8], 4)


    tile_shape_config([1024,1024], 8)
    tile_shape_config([1024,32], 8)
    tile_shape_config([1024,16], 8)
    tile_shape_config([1024,8], 8)

    tile_shape_config([128,1024], 8)
    tile_shape_config([64,32], 8)
    tile_shape_config([32,16], 8)
    tile_shape_config([32,8], 8)

    tile_shape_config([1024,1024], 6)
    tile_shape_config([1024,32], 6)
    tile_shape_config([1024,16], 6)
    tile_shape_config([1024,8], 6)
    tile_shape_config([1024,1], 6)

    tile_shape_config([1000,1024], 6)
    tile_shape_config([1000,32], 6)
    tile_shape_config([1000,16], 6)
    tile_shape_config([1000,8], 6)


    tile_shape_config([64, 32], 8)
    tile_shape_config([128, 32], 8)
    tile_shape_config([64, 64], 8)
    tile_shape_config([512, 8], 8)
    tile_shape_config([511, 7], 8)

if __name__ == '__main__':
    main()
